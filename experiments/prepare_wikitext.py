"""Prepare wikitext-103 for FIP-Net: download -> BPE tokenizer -> token shards.

Run on Kaggle (has internet + `datasets` preinstalled):
    python experiments/prepare_wikitext.py --vocab 8192 --max-tokens 100_000_000

Outputs into --out-dir (default data/):
    train_tokens.npy / val_tokens.npy  (uint16, works for V <= 65535)
    wikitext_tokenizer.json
    bytes_mat.pt                       (V x n_bytes byte rows, GPU-ready)
"""
import argparse
import os
import subprocess
import sys

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n-bytes", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=100_000_000)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # datasets' PyExtensionType usage breaks with pyarrow>=15; downgrade
    # to the known-good pair before importing (no-op if already satisfied)
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "numpy==1.26.4", "datasets==2.19.1", "pyarrow==14.0.2",
                    "fsspec==2024.3.0"],
                   check=False)
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    print("downloading wikitext-103-raw-v1 ...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    tok_path = os.path.join(args.out_dir, "wikitext_tokenizer.json")
    if not os.path.exists(tok_path):
        print("training BPE tokenizer ...", flush=True)
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab, special_tokens=["<|pad|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        def train_iter():
            for ex in ds["train"]:
                t = ex["text"]
                if t.strip():
                    yield t
        tok.train_from_iterator(train_iter(), trainer, length=len(ds["train"]))
        tok.save(tok_path)
    tok = Tokenizer.from_file(tok_path)
    V = tok.get_vocab_size()
    print(f"vocab: {V}", flush=True)

    # byte rows for the procedural input coder
    rows = np.zeros((V, args.n_bytes), dtype=np.int64)
    for i in range(V):
        s = (tok.id_to_token(i) or "").replace("\u0120", " ").replace("\u010a", "\n")
        b = s.encode("utf-8")[: args.n_bytes]
        rows[i, :len(b)] = list(b)
    import torch
    torch.save(torch.from_numpy(rows), os.path.join(args.out_dir, "bytes_mat.pt"))

    def encode_split(split, cap):
        ids = []
        for ex in ds[split]:
            t = ex["text"]
            if t.strip():
                ids.extend(tok.encode(t).ids)
            if len(ids) >= cap:
                break
        return np.asarray(ids[:cap], dtype=np.uint16)

    print("tokenizing train ...", flush=True)
    tr = encode_split("train", args.max_tokens)
    print("tokenizing val ...", flush=True)
    va = encode_split("test", args.val_tokens)
    np.save(os.path.join(args.out_dir, "train_tokens.npy"), tr)
    np.save(os.path.join(args.out_dir, "val_tokens.npy"), va)
    print(f"done: train {len(tr)/1e6:.1f}M tokens, val {len(va)/1e3:.0f}K, "
          f"V={V}, n_bytes={args.n_bytes}", flush=True)

if __name__ == "__main__":
    main()
