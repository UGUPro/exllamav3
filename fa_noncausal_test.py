import torch, math
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda:0"
dt = torch.float16

# Draft-model-like shapes: head_dim 128, GQA 64q/8kv, small q-block appended to a cache
B, Hq, Hkv, D = 1, 8, 2, 128
q_len = 16          # DFlash block
cache_len = 512     # existing context already in cache
window = 2048       # sliding window (>= total, so effectively full)

def ref_attn(q, k_full, v_full, causal, wl, wr):
    # q: [B,q_len,Hq,D]  k_full/v_full: [B,T,Hkv,D]   GQA expand
    # window_size=(wl,wr): key j allowed if qpos-wl <= j <= qpos+wr ; -1 = unbounded
    B, Lq, Hq, D = q.shape
    T = k_full.shape[1]
    g = Hq // k_full.shape[2]
    k = k_full.repeat_interleave(g, dim=2)
    v = v_full.repeat_interleave(g, dim=2)
    qs = q.permute(0,2,1,3).float()
    ks = k.permute(0,2,1,3).float()
    vs = v.permute(0,2,1,3).float()
    scores = torch.matmul(qs, ks.transpose(-1,-2)) / math.sqrt(D)
    qpos = torch.arange(T-Lq, T, device=q.device).view(1,1,Lq,1)
    kpos = torch.arange(T, device=q.device).view(1,1,1,T)
    mask = torch.zeros(1,1,Lq,T, device=q.device, dtype=torch.bool)
    if causal:
        mask |= (kpos > qpos)
    if wl >= 0:
        mask |= (kpos < qpos - wl)
    if wr >= 0:
        mask |= (kpos > qpos + wr)
    scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vs)
    return o.permute(0,2,1,3)  # [B,Lq,Hq,D]

# (label, causal, window_left, window_right) -- the cases DFlash actually uses
cases = [
    ("causal q-block (sanity)",      True,  -1, -1),
    ("sliding non-causal (2048,0)",  False, window, 0),
    ("full non-causal (-1,-1)",      False, -1, -1),
]
for label, causal, wl, wr in cases:
    # Build a paged-style cache big enough; flash_attn_with_kvcache appends q's k/v
    total = cache_len + q_len
    k_cache = torch.zeros(B, total, Hkv, D, dtype=dt, device=dev)
    v_cache = torch.zeros(B, total, Hkv, D, dtype=dt, device=dev)
    k_cache[:, :cache_len] = torch.randn(B, cache_len, Hkv, D, dtype=dt, device=dev)
    v_cache[:, :cache_len] = torch.randn(B, cache_len, Hkv, D, dtype=dt, device=dev)
    q = torch.randn(B, q_len, Hq, D, dtype=dt, device=dev)
    k = torch.randn(B, q_len, Hkv, D, dtype=dt, device=dev)
    v = torch.randn(B, q_len, Hkv, D, dtype=dt, device=dev)
    cache_seqlens = torch.full((B,), cache_len, dtype=torch.int32, device=dev)

    out = flash_attn_with_kvcache(
        q=q, k=k, v=v, k_cache=k_cache, v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        window_size=(wl, wr),
        causal=causal,
    )
    # reference over the now-updated cache
    k_full = k_cache[:, :total].clone()
    v_full = v_cache[:, :total].clone()
    ref = ref_attn(q, k_full, v_full, causal, wl, wr)
    err = (out.float() - ref.float()).abs()
    rel = (err.max() / (ref.float().abs().max() + 1e-6)).item()
    print(f"{label:32s} causal={causal} win=({wl},{wr}): "
          f"max_abs_err={err.max().item():.4f} rel={rel:.4f} "
          f"{'OK' if rel < 0.05 else 'FAIL <<<'}")
