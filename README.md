# FIP-Net: Fast Isometric Periodic Network

A norm-free, table-free LM architecture: periodic wave activation on FMA
(no SFU), zero vocabulary parameters (procedural byte-harmonic input +
low-rank phase head), variance-preserving phase skip instead of plain
residual, RoPE attention, and an exact-isometry two-parameter (gamma, beta)
calibration.

## Results (TinyStories, BPE-2048, 30k steps unless noted)

| Model | Params | Vocab params | acc(all) / acc(last) |
|---|---|---|---|
| ATT-Medium-F (D192, L8, na2, h4, rope+gated, iso-prior) 60k cosine | 0.68M | 0 | **43.3 / 44.4** |
| llama-equal (RMSNorm+RoPE+SwiGLU) 30k | 2.19M | 655K | 44.5 / — |
| ATT-iso2 (D152, na2, h2, L8) | 0.49M | 0 | 40.4 |
| ATT-Medium-F 30k | 0.68M live | 0 | 41.4 / 42.8 |
| ATT-F-iso (D164, L8, na1) | 0.425M | 0 | 38.2 |
| llama-small (tied) 30k | 0.426M | 131K | 37.2 |
| ATT-embed (real embedding table) | 0.49M | 242K | 27.9 |
| Classic A (emb+ReLU+softmax) | 1.67M | 1.15M | 30.9 |

Key: at ~0.426M, ATT with **zero vocab params** beats llama-small; the
embedding-table ablation loses by 10 p.p. to the procedural coder.

## Layout

- `fipnet/` — core: activation (parabolic/triangular waves, analytic
  backward), model (AttLM, LowRankPhaseBottleneck), CUDA-graph step
- `experiments/` — attention_v3 (champion LM), mini_llama (llama baseline),
  distill (two-half + end-to-end Top-K/DKD v3 protocols), ddp_train
  (Kaggle T4x2), ablations, stage-0 (gram codebook) ablation
- `tests/` — 27 unit tests (activation gradients, model, decode)
- `kaggle/FIP_T4x2.md` — dataset upload list + notebook cells + scale ladder

## Quick start

```bash
python -m pytest tests/ -q
python experiments/attention_v3.py            # train champion config
python experiments/ddp_train.py --world 2     # 2-GPU DDP (FileStore, no TCP)
```

## Notes

- Checkpoints are atomic (tmp + os.replace) everywhere.
- Eval calls torch.cuda.empty_cache() - eval transients must never sit in
  the caching allocator (WDDM 4GB-card spillover lesson).
- Memory hygiene: teacher logits in bf16, chunked full-vocab KD.
