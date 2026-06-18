"""ROCm flash-attention gating.

On ROCm there are two importable flash-attn backends, and they behave very
differently on RDNA (e.g. gfx1151 / Strix Halo):

  * aiter **Triton** backend (the default `pip install` on ROCm): BROKEN here.
    Invoking ``flash_attn_with_kvcache`` corrupts GPU/queue state that other
    kernels depend on -> garbage output, cross-request KV contamination, and on
    some shapes a hard GPU fault (HSA_STATUS_ERROR_EXCEPTION 0x1016). Its own
    numerics are fine; the damage is a side effect of running the kernel.
  * **Composable Kernel (CK)** backend (built from ROCm/flash-attention with
    ``FLASH_ATTENTION_TRITON_AMD_ENABLE`` unset): correct and faster. It ships
    the compiled ``flash_attn_2_cuda`` extension; the Triton fallback does not.

So the default policy on ROCm is: use flash-attn only when the CK backend is
present (detected via an importable ``flash_attn_2_cuda``); otherwise fall
through to the correct triton_paged / SDPA path. ``EXLLAMA_ROCM_ALLOW_FLASH``
overrides this: ``1`` forces flash on (e.g. to benchmark the Triton backend),
``0`` forces it off.

Non-ROCm (CUDA) builds are never gated here.
"""
import os
import torch


def rocm_flash_disabled() -> bool:
    if torch.version.hip is None:
        return False
    override = os.environ.get("EXLLAMA_ROCM_ALLOW_FLASH")
    if override == "1":
        return False
    if override == "0":
        return True
    # Auto: allow only the CK backend (ships flash_attn_2_cuda); the Triton
    # fallback lacks it and is unsafe on RDNA.
    try:
        import flash_attn_2_cuda  # noqa: F401
        return False
    except (ImportError, ModuleNotFoundError):
        return True
