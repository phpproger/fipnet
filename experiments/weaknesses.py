"""Weakness audit of ATT (FIP) from all session evidence, with fixes.

Evidence-driven weakness inventory:

W1  Fixed LR, no schedule. The 30k medium-F trajectory was still climbing
    (+0.4..0.5 p.p. per 5k at step 30k: 36.3, 38.4, 39.5, 40.3, 41.0, 41.4).
    Constant-LR leaves 1-3+ p.p. on the table vs cosine decay + warmup.
    -> FIX E1: cosine schedule (tensor-lr inside capturable fused Adam so
       CUDA-graph replay sees updates), warmup 500.

W2  Attention capacity at fixed budget. Champion iso config spends the whole
    0.426M on n_attn=1, heads=1. medium-F (0.78M, n_attn=2, heads=4) beat
    iso by +3.2 p.p. - but that confuses width/depth/attention. Is one
    attention pattern the bottleneck at EQUAL budget?
    -> FIX E2: ATT-F-iso2: D=144, n_attn=2, heads=2, L=8, 0.428M (vs
       llama-small 0.426M): same total params, 2x attention patterns.
       FLOPs of attention at D=144 x2 heads = same as D=192 x1.

W3  Head bottleneck K=32 (Welch ~K^2=1024 well-separated classes < V=16384).
    Real only for large vocab; V=2048 is fine (gram loss 0.033). Defer to
    the Kaggle 16k runs: try K=64 (+65K params).

W4  Depth scaling / isometry drift (depth-14 from-scratch collapse).
    Soft iso-prior (lambda=1e-3) works for D=192/L8. For L14 test freezing
    beta/psi after warmup (hard anchor) vs the prior.
    -> FIX E3 (later, with the Kaggle depth-14 runs): hard-freeze variant.

W5  Training length. 30k is short; llama-equal hit 44.5% at 30k with 2.19M.
    medium-F at 30k = 41.4% with 0.78M and still climbing.
    -> FIX E1': 60k cosine runs on BOTH sides for the honest long-run table:
       medium-F-60k-cosine (ATT) vs llama-small-60k-cosine (llama).

W6  Attention pattern diversity: heads=1 in iso. E2 also answers this
    (heads=2).

W7  Large-vocab byte truncation (n_bytes=8) for BPE-16384 long tokens:
    increase n_bytes to 12-16 on the 16k experiments (Kaggle).

Priorities: E1 (schedule) > E2 (attention split) > E5 (length, combined
with E1 in the same run) > W3/W4/W7 on the large-vocab Kaggle runs.
"""
