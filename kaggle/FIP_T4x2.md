# FIP-Net on Kaggle T4 ×2 — run guide

## 0. What to upload as a Kaggle dataset (one-time)

| File | Size | Source |
|---|---|---|
| `train2048.npy` | 999 MB | `C:\Users\ignat\PycharmProjects\future\tinystories_full_bpe2048\` |
| `val2048.npy` | 10 MB | same dir |
| `bytes_mat_2048.pt` | 65 KB | `data/bytes_mat_2048.pt` (portable byte rows, tokenizer NOT needed) |
| `codebook_2048_fourier32.pt` | 65 KB | `data/codebook_2048_fourier32.pt` |
| code zip | small | `fipnet/`, `experiments/`, `tests/` (archive repo as `fipcode.zip`) |

Name the dataset `fipdata`. Kaggle GPU sessions mount it at `/kaggle/input/fipdata/`.

## 1. Notebook cell: unpack code

```python
import zipfile, os
os.makedirs("/kaggle/working/fip", exist_ok=True)
with zipfile.ZipFile("/kaggle/input/fipdata/fipcode.zip") as z:
    z.extractall("/kaggle/working/fip")
os.chdir("/kaggle/working/fip")
# copy the small aux files next to the big npy shards
import shutil
DATA = "/kaggle/working/data"
os.makedirs(DATA, exist_ok=True)
shutil.copy("/kaggle/input/fipdata/train2048.npy", DATA)
shutil.copy("/kaggle/input/fipdata/val2048.npy", DATA)
shutil.copy("/kaggle/input/fipdata/bytes_mat_2048.pt", DATA)
shutil.copy("/kaggle/input/fipdata/codebook_2048_fourier32.pt", DATA)
```

## 2. Quick sanity: single GPU, small model, measure speed

```python
!python experiments/ddp_train.py --world 1 --bench \
    --dim 256 --depth 10 --n-attn 2 --heads 2 --steps 300 \
    --data-dir /kaggle/working/data --bytes-mat data/bytes_mat_2048.pt \
    --codebook data/codebook_2048_fourier32.pt \
    --ckpt /kaggle/working/ck_sanity.pt
```

Read `bench` lines: ms/step/rank. Also run once with `--amp` and once without;
keep whichever is faster (T4 fp16 tensor cores usually win).

## 3. Main run: DDP over 2×T4

```python
!torchrun --standalone --nproc_per_node=2 experiments/ddp_train.py \
    --dim 256 --depth 10 --n-attn 2 --heads 2 --steps 60000 --batch 256 \
    --data-dir /kaggle/working/data --bytes-mat data/bytes_mat_2048.pt \
    --codebook data/codebook_2048_fourier32.pt \
    --ckpt /kaggle/working/ck_largeA.pt 2>&1 | tee largeA.log
```

* Checkpoints land in `/kaggle/working/` (persist across notebook re-runs within
  a session; for cross-session persistence save them to a Kaggle output dataset).
* If the session dies: rerun the same cell — it resumes from the checkpoint
  (rank 0 file, atomic; safe against kill at any moment).

## 4. Recommended scale ladder (iso-battle continuation)

| Config | dim | depth | n_attn | heads | Params | vs |
|---|---|---|---|---|---|---|
| Large-A | 256 | 10 | 2 | 2 | ~1.39M | llama-equal 2.19M |
| Large-B | 320 | 10 | 2 | 4 | ~2.13M | llama-equal 2.19M (iso!) |
| Large-C | 256 | 14 | 2 | 2 | ~1.65M | depth-14 anti-collapse claim |

* Large-B is the headline: **iso-params with the 44.46% llama-equal teacher**.
* Same machinery later works for BPE-16384: swap `train16384.npy`/`val16384.npy`,
  `bytes_mat_16384.pt`, `codebook_16384.pt` (needs the 16k tokenizer on the
  packing machine only).

## 4b. Wikitext-103 path (no TinyStories upload needed)

Clone + prepare + train the ~10M model:

```python
!git clone https://github.com/phpproger/fipnet.git /kaggle/working/fipnet
%cd /kaggle/working/fipnet
!pip -q install datasets
!python experiments/prepare_wikitext.py --vocab 8192 --n-bytes 12     --max-tokens 100000000 --out-dir data
# (~15-25 min: download + BPE-8192 + tokenize 100M tokens)

!torchrun --standalone --nproc_per_node=2 experiments/ddp_train.py     --dim 640 --depth 14 --n-attn 2 --heads 4     --train-file train_tokens.npy --val-file val_tokens.npy     --n-bytes 12 --vocab 8192     --steps 50000 --batch 512 --lr 3e-4     --data-dir data --bytes-mat data/bytes_mat.pt     --ckpt /kaggle/working/ck_wiki10m.pt
```

Config sheet (V=8192, n_bytes=12, code_dim=32):
  D640 L14 na2 h4 = 9.87M  <- headline 10M
  D576 L14 na3 h4 = 9.67M  (3 attention patterns)
  D512 L16 na2 h4 = 6.84M
VRAM at B=512x64: ~4-6GB per T4 (16GB) - headroom for --window 128
(attention FLOPs x4, rope/mask now handle any window).

## 5. Gotchas

* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the scripts set it via
  env in our bat files; on Kaggle add `%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  as the first notebook line.
* After every eval the code calls `torch.cuda.empty_cache()` — eval transients
  must never sit in the allocator cache (learned on the 3050 the hard way).
* Kaggle sessions have 12h GPU cap: pick steps so the run finishes or rely on
  resume across two sessions (checkpoint is the only state needed).
* `--n-attn 2` students need `--depth >= 2`; distillation stage-1 requires
  `--comp-blocks >= n_attn` (all attention inside the compressor).
