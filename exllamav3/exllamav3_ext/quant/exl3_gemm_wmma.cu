#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <array>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_gemm_wmma.cuh"

// Fused dequant + matmul for ROCm/RDNA3.5 (gfx11), using the native WMMA
// intrinsic instead of NVIDIA tensor-core PTX. Computes C[M,N] = A[M,K] @ W[K,N]
// where W is the exl3 trellis-quantized weight. The 16x128 weight block is
// decoded into shared memory exactly as reconstruct.cu does (so the decode is
// the already-validated path), but kept on-chip and consumed directly by WMMA,
// avoiding the full-weight round trip through global memory that the
// reconstruct + hgemm fallback incurs.

#ifdef USE_ROCM

using half16 = __attribute__((ext_vector_type(16))) __fp16;
using float8 = __attribute__((ext_vector_type(8))) float;

// RDNA3 wave32 WMMA fragment layout (empirically verified):
//   A: lane l holds row (l%16), all 16 K           -> a_frag[k] = A[m+(l%16)][k0+k]
//   B: lane l holds col (l%16), all 16 K           -> b_frag[k] = W[k0+k][n+(l%16)]
//   C: c_frag[i] = C[2*i + (l/16)][l%16]
template <int K, int cb, bool c_fp32>
__global__ __launch_bounds__(256)
void exl3_gemm_wmma_kernel
(
    const half* __restrict__ A,             // [M, K_dim]
    const uint16_t* __restrict__ trellis,   // [K_dim/16, N/16, packed_size]
    void* __restrict__ C,                   // [M, N]
    int M,
    int K_dim,
    int N,
    int packed_blocks_n                     // N / 16
)
{
    constexpr int packed_size = 256 * K / 16;   // uint16s per 16x16 block

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;          // 0..7, this warp's N-subtile within the 128-wide tile
    int n_tile = blockIdx.x;       // 128-wide N tile
    int m_tile = blockIdx.y * 16;  // M base
    int n_base = n_tile * 128;
    int kblocks = K_dim / 16;

    __shared__ uint32_t s_packed[8][packed_size / 2];
    __shared__ half2 tile[16][8][8];   // [k_row][n_block][n_half2] row-major W block

    float8 c_frag = {};

    for (int kb = 0; kb < kblocks; ++kb)
    {
        // Load packed 16x128 weight block for (kb, n_tile)
        const uint16_t* g_packed = trellis + ((size_t)(kb * packed_blocks_n) + n_tile * 8) * packed_size;
        if (t < packed_size)
            ((int4*) s_packed)[t] = ((const int4*) g_packed)[t];
        __syncthreads();

        // Decode this warp's 16x16 N-block and shuffle to row major (reconstruct.cu logic)
        FragB frag[2];
        dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);
        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);
        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int c1 = c0 + 4;
            tile[r0][warp_id][c0] = m0;
            tile[r1][warp_id][c0] = m1;
            tile[r2][warp_id][c0] = m2;
            tile[r3][warp_id][c0] = m3;
            tile[r0][warp_id][c1] = m4;
            tile[r1][warp_id][c1] = m5;
            tile[r2][warp_id][c1] = m6;
            tile[r3][warp_id][c1] = m7;
        }
        __syncthreads();

        // WMMA: this warp computes a 16(M) x 16(N) output subtile for N-block warp_id
        half16 a_frag, b_frag;
        int ar = m_tile + (lane_id % 16);
        const half* a_row = A + (size_t) ar * K_dim + kb * 16;
        const half* w_row_base = (const half*) &tile[0][warp_id][0];
        #pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            a_frag[k] = (ar < M) ? (__fp16) __half2float(a_row[k]) : (__fp16) 0;
            // tile[k][warp_id] is 16 contiguous halfs (8 half2); stride between k rows is 8*8 half2 = 64 half2 = 128 half
            b_frag[k] = (__fp16) __half2float(((const half*) &tile[k][warp_id][0])[lane_id % 16]);
        }
        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store: c_frag[i] -> C[m_tile + 2*i + lane/16][n_base + warp_id*16 + lane%16]
    int col = n_base + warp_id * 16 + (lane_id % 16);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        int r = m_tile + 2 * i + (lane_id / 16);
        if (r < M)
        {
            if constexpr (c_fp32)
                ((float*) C)[(size_t) r * N + col] = c_frag[i];
            else
                ((half*) C)[(size_t) r * N + col] = __float2half(c_frag[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused dequant + GEMV for the decode path (small M). Each 16x128 weight block
// is decoded exactly once (like reconstruct), accumulated straight into the
// output with split-K + atomics, avoiding the dense-weight global round trip.
// No WMMA (which wastes 15/16 at M=1); a plain scalar dot is bandwidth-optimal.
// ---------------------------------------------------------------------------
#define EXL3_GEMV_MAXM 8

template <int K, int cb>
__global__ __launch_bounds__(256)
void exl3_gemv_fused_kernel
(
    const half* __restrict__ A,             // [M, K_dim]
    const uint16_t* __restrict__ trellis,   // [K_dim/16, N/16, packed_size]
    float* __restrict__ C,                  // [M, N] fp32, pre-zeroed
    int M,
    int K_dim,
    int N,
    int packed_blocks_n,                    // N / 16
    int kb_per_block
)
{
    constexpr int packed_size = 256 * K / 16;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;          // 0..7, this warp's 16-N subtile
    int n_tile = blockIdx.x;       // 128-wide N tile
    int n_base = n_tile * 128;
    int kb0 = blockIdx.y * kb_per_block;
    int kb1 = min(kb0 + kb_per_block, K_dim / 16);

    __shared__ uint32_t s_packed[8][packed_size / 2];
    __shared__ half2 tile[16][8][8];

    float acc[EXL3_GEMV_MAXM];
    #pragma unroll
    for (int m = 0; m < EXL3_GEMV_MAXM; ++m) acc[m] = 0.0f;

    for (int kb = kb0; kb < kb1; ++kb)
    {
        const uint16_t* g_packed = trellis + ((size_t)(kb * packed_blocks_n) + n_tile * 8) * packed_size;
        if (t < packed_size)
            ((int4*) s_packed)[t] = ((const int4*) g_packed)[t];
        __syncthreads();

        // Decode this warp's 16x16 N-block to row-major shared (reconstruct.cu logic)
        FragB frag[2];
        dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);
        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);
        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = (lane_id % 4) * 2; int r1 = r0 + 1; int r2 = r0 + 8; int r3 = r0 + 9;
            int c0 = lane_id / 8; int c1 = c0 + 4;
            tile[r0][warp_id][c0] = m0; tile[r1][warp_id][c0] = m1;
            tile[r2][warp_id][c0] = m2; tile[r3][warp_id][c0] = m3;
            tile[r0][warp_id][c1] = m4; tile[r1][warp_id][c1] = m5;
            tile[r2][warp_id][c1] = m6; tile[r3][warp_id][c1] = m7;
        }
        __syncthreads();

        // Accumulate: lane (0..15) owns N column (n_base + warp_id*16 + lane)
        if (lane_id < 16)
        {
            int kk = kb * 16;
            #pragma unroll
            for (int k = 0; k < 16; ++k)
            {
                float wv = __half2float(((const half*) &tile[k][warp_id][0])[lane_id]);
                #pragma unroll
                for (int m = 0; m < EXL3_GEMV_MAXM; ++m)
                    if (m < M) acc[m] += __half2float(A[(size_t) m * K_dim + kk + k]) * wv;
            }
        }
        __syncthreads();
    }

    if (lane_id < 16)
    {
        int col = n_base + warp_id * 16 + lane_id;
        #pragma unroll
        for (int m = 0; m < EXL3_GEMV_MAXM; ++m)
            if (m < M) atomicAdd(&C[(size_t) m * N + col], acc[m]);
    }
}

#define __g(i, cb) exl3_gemv_fused_kernel<i, cb>
constexpr auto exl3_gemv_fused_instances = std::array
{
    __g(1, 0), __g(2, 0), __g(3, 0), __g(4, 0), __g(5, 0), __g(6, 0), __g(7, 0), __g(8, 0),
    __g(1, 1), __g(2, 1), __g(3, 1), __g(4, 1), __g(5, 1), __g(6, 1), __g(7, 1), __g(8, 1),
    __g(1, 2), __g(2, 2), __g(3, 2), __g(4, 2), __g(5, 2), __g(6, 2), __g(7, 2), __g(8, 2)
};
#undef __g

#define __(i, cb) exl3_gemm_wmma_kernel<i, cb, false>, exl3_gemm_wmma_kernel<i, cb, true>
constexpr auto exl3_gemm_wmma_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

#endif // USE_ROCM

void exl3_gemm_wmma
(
    at::Tensor a,
    at::Tensor trellis,
    at::Tensor c,
    int64_t K,
    bool mcg,
    bool mul1
)
{
#ifdef USE_ROCM
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int M = a.size(0);
    int K_dim = a.size(1);
    int N = c.size(1);
    int packed_blocks_n = trellis.size(1);

    TORCH_CHECK(N % 128 == 0, "exl3_gemm_wmma: N must be divisible by 128");
    TORCH_CHECK(K_dim % 16 == 0, "exl3_gemm_wmma: K must be divisible by 16");
    TORCH_CHECK(a.dtype() == at::kHalf, "exl3_gemm_wmma: a must be fp16");
    bool c_fp32 = (c.dtype() == at::kFloat);
    TORCH_CHECK(c_fp32 || c.dtype() == at::kHalf, "exl3_gemm_wmma: c must be fp16 or fp32");

    int cbi = (int) K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    int idx = cbi * 2 + (c_fp32 ? 1 : 0);

    dim3 gridDim(N / 128, (M + 15) / 16);
    dim3 blockDim(256);
    auto kernel = exl3_gemm_wmma_instances[idx];
    kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (const half*) a.data_ptr(),
        (const uint16_t*) trellis.data_ptr(),
        c.data_ptr(),
        M, K_dim, N, packed_blocks_n
    );
    cuda_check(cudaPeekAtLastError());
#else
    TORCH_CHECK(false, "exl3_gemm_wmma is only available on ROCm");
#endif
}

void exl3_gemv_fused
(
    at::Tensor a,
    at::Tensor trellis,
    at::Tensor c,        // [M, N] fp32, must be pre-zeroed (accumulated via atomics)
    int64_t K,
    bool mcg,
    bool mul1
)
{
#ifdef USE_ROCM
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int M = a.size(0);
    int K_dim = a.size(1);
    int N = c.size(1);
    int packed_blocks_n = trellis.size(1);

    TORCH_CHECK(N % 128 == 0, "exl3_gemv_fused: N must be divisible by 128");
    TORCH_CHECK(K_dim % 16 == 0, "exl3_gemv_fused: K must be divisible by 16");
    TORCH_CHECK(M <= EXL3_GEMV_MAXM, "exl3_gemv_fused: M too large");
    TORCH_CHECK(a.dtype() == at::kHalf, "exl3_gemv_fused: a must be fp16");
    TORCH_CHECK(c.dtype() == at::kFloat, "exl3_gemv_fused: c must be fp32");

    int n_tiles = N / 128;
    int k_blocks = K_dim / 16;
    // Pick split-K so total blocks land in a healthy range for occupancy.
    int target_blocks = 512;
    int nsplit = (target_blocks + n_tiles - 1) / n_tiles;
    if (nsplit < 1) nsplit = 1;
    if (nsplit > k_blocks) nsplit = k_blocks;
    int kb_per_block = (k_blocks + nsplit - 1) / nsplit;
    nsplit = (k_blocks + kb_per_block - 1) / kb_per_block;

    int cbi = (int) K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;

    dim3 gridDim(n_tiles, nsplit);
    dim3 blockDim(256);
    auto kernel = exl3_gemv_fused_instances[cbi];
    kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (const half*) a.data_ptr(),
        (const uint16_t*) trellis.data_ptr(),
        (float*) c.data_ptr(),
        M, K_dim, N, packed_blocks_n, kb_per_block
    );
    cuda_check(cudaPeekAtLastError());
#else
    TORCH_CHECK(false, "exl3_gemv_fused is only available on ROCm");
#endif
}
