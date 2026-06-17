#pragma once
#include <ATen/Tensor.h>

// Fused dequant + matmul using RDNA3 WMMA (ROCm only). C[M,N] = A[M,K] @ W[K,N]
// where W is the exl3 trellis. a: [M,K] fp16; trellis: [K/16, N/16, 256*K_bits/16]
// uint16; c: [M,N] fp16 or fp32 (preallocated).
void exl3_gemm_wmma(at::Tensor a, at::Tensor trellis, at::Tensor c, int64_t K, bool mcg, bool mul1);

// Fused dequant + GEMV for small M (decode). c: [M,N] fp32, pre-zeroed.
void exl3_gemv_fused(at::Tensor a, at::Tensor trellis, at::Tensor c, int64_t K, bool mcg, bool mul1);
