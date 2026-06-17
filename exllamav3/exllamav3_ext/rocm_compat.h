#pragma once

// ROCm/HIP compatibility shim, force-included for HIP device compilation.
//
// The code base uses CUDA warp-synchronous primitives with explicit 32-bit
// masks (always the full-warp 0xffffffff). HIP's masked *_sync builtins
// static_assert on a 64-bit mask, so we build with
// -DHIP_DISABLE_WARP_SYNC_BUILTINS (which removes them) and map the *_sync
// names onto HIP's legacy mask-less warp builtins here. Dropping the mask is
// correct because every call site uses the full-warp mask.

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HIP's legacy __shfl* builtins provide overloads for int/float/double/long but
// NOT for __half / __half2 (CUDA's *_sync intrinsics do). Without these, a
// __half2 argument silently converts to a scalar and the shuffled data is
// corrupted. Route the *_sync names through exl3_shfl_* wrappers: a generic
// template delegates to the base builtin (int/float/...), with bit-preserving
// __half / __half2 overloads. This avoids overloading the global __shfl* names
// (which would make int/float calls ambiguous).
#define EXL3_DEFINE_SHFL(WRAP, BASE)                                                     \
    template <typename T>                                                               \
    __device__ __forceinline__ T WRAP(T var, int a, int width = warpSize)               \
    { return BASE(var, a, width); }                                                     \
    __device__ __forceinline__ __half2 WRAP(__half2 var, int a, int width = warpSize)    \
    {                                                                                    \
        unsigned int u; __builtin_memcpy(&u, &var, sizeof(u));                           \
        u = BASE(u, (unsigned int) a, width);                                           \
        __half2 r; __builtin_memcpy(&r, &u, sizeof(r));                                  \
        return r;                                                                        \
    }                                                                                    \
    __device__ __forceinline__ __half WRAP(__half var, int a, int width = warpSize)      \
    {                                                                                    \
        unsigned short s; __builtin_memcpy(&s, &var, sizeof(s));                         \
        unsigned int u = BASE((unsigned int) s, (unsigned int) a, width);               \
        s = (unsigned short) u; __half r; __builtin_memcpy(&r, &s, sizeof(r));           \
        return r;                                                                        \
    }
EXL3_DEFINE_SHFL(exl3_shfl,      __shfl)
EXL3_DEFINE_SHFL(exl3_shfl_up,   __shfl_up)
EXL3_DEFINE_SHFL(exl3_shfl_down, __shfl_down)
EXL3_DEFINE_SHFL(exl3_shfl_xor,  __shfl_xor)
#undef EXL3_DEFINE_SHFL

#define __shfl_sync(mask, var, ...)       exl3_shfl(var, __VA_ARGS__)
#define __shfl_up_sync(mask, var, ...)    exl3_shfl_up(var, __VA_ARGS__)
#define __shfl_down_sync(mask, var, ...)  exl3_shfl_down(var, __VA_ARGS__)
#define __shfl_xor_sync(mask, var, ...)   exl3_shfl_xor(var, __VA_ARGS__)
#define __ballot_sync(mask, pred)         __ballot(pred)
#define __any_sync(mask, pred)            __any(pred)
#define __all_sync(mask, pred)            __all(pred)
#define __syncwarp(...)                   do {} while (0)
#ifndef __activemask
#define __activemask()                    0xffffffffu
#endif

// HIP lacks the scalar float->bf16 rounding-mode intrinsics; round-to-nearest
// is fine for all uses here.
#define __float2bfloat16_rn __float2bfloat16
#define __float2bfloat16_rz __float2bfloat16

// HIP has no __nanosleep; the only uses are backoff spins, so a short sleep is
// fine. The argument (nanoseconds) is ignored.
#define __nanosleep(ns) __builtin_amdgcn_s_sleep(1)

// CUDA kernel-parameter attribute with no HIP equivalent; drop it.
#define __grid_constant__

// HIP (ROCm 7.x) does not provide the __half2 overloads of __hmin2/__hmax2.
__device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b)
{
    __half al = __low2half(a), ah = __high2half(a);
    __half bl = __low2half(b), bh = __high2half(b);
    return __halves2half2(__hlt(al, bl) ? al : bl, __hlt(ah, bh) ? ah : bh);
}
__device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b)
{
    __half al = __low2half(a), ah = __high2half(a);
    __half bl = __low2half(b), bh = __high2half(b);
    return __halves2half2(__hgt(al, bl) ? al : bl, __hgt(ah, bh) ? ah : bh);
}

#endif // USE_ROCM / __HIP_PLATFORM_AMD__
