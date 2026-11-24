"""Prepare wikitext-103 for FIP-Net: HF parquet -> BPE -> token shards.

Dependency-light by design: NO datasets/pandas/pyarrow (numpy-ABI hell
across Kaggle images). Downloads the HF parquet shards directly and reads
them with polars (pure rust wheel).

    python experiments/prepare_wikitext.py --vocab 8192 --n-bytes 12 \
        --max-tokens 100000000 --out-dir data

Outputs: train_tokens.npy / val_tokens.npy (uint16), wikitext_tokenizer.json,
bytes_mat.pt (V x n_bytes rows).
"""
import argparse
import os
import subprocess
import sys
import urllib.request

# The original .zip is dead on every mirror (smerity 403, S3 403 - see
# huggingface/tokenizers#1683). HF now ships parquet shards.
BASE = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1")
SHARDS = {
    "train": [f"{BASE}/train-00000-of-00002.parquet",
              f"{BASE}/train-00001-of-00002.parquet"],
    "test": [f"{BASE}/test-00000-of-00001.parquet"],
}


def fetch_split(split, out_dir):
    import polars as pl
    lines = []
    for i, url in enumerate(SHARDS[split]):
        path = os.path.join(out_dir, f"{split}-{i}.parquet")
        if not os.path.exists(path):
            print(f"downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        df = pl.read_parquet(path)
        lines.extend(t for t in df["text"].to_list() if t.strip())
        print(f"  {split}[{i}]: {df.height} rows", flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n-bytes", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=100_000_000)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars"],
                   check=False)
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    train_lines = fetch_split("train", args.out_dir)
    test_lines = fetch_split("test", args.out_dir)
    print(f"train lines: {len(train_lines)}, test lines: {len(test_lines)}",
          flush=True)

    tok_path = os.path.join(args.out_dir, "wikitext_tokenizer.json")
    if not os.path.exists(tok_path):
        print("training BPE tokenizer ...", flush=True)
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab, special_tokens=["<|pad|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(iter(train_lines), trainer)
        tok.save(tok_path)
    tok = Tokenizer.from_file(tok_path)
    V = tok.get_vocab_size()
    print(f"vocab: {V}", flush=True)

    import numpy as np
    import torch
    rows = np.zeros((V, args.n_bytes), dtype=np.int64)
    for i in range(V):
        s = (tok.id_to_token(i) or "").replace("\u0120", " ").replace("\u010a", "\n")
        b = s.encode("utf-8")[: args.n_bytes]
        rows[i, :len(b)] = list(b)
    torch.save(torch.from_numpy(rows), os.path.join(args.out_dir, "bytes_mat.pt"))

    def encode(lines_, cap):
        ids = []
        for line in lines_:
            ids.extend(tok.encode(line).ids)
            if len(ids) >= cap:
                break
        return np.asarray(ids[:cap], dtype=np.uint16)

    print("tokenizing train ...", flush=True)
    tr = encode(train_lines, args.max_tokens)
    print("tokenizing test ...", flush=True)
    va = encode(test_lines, args.val_tokens)
    np.save(os.path.join(args.out_dir, "train_tokens.npy"), tr)
    np.save(os.path.join(args.out_dir, "val_tokens.npy"), va)
    print(f"done: train {len(tr)/1e6:.1f}M tokens, val {len(va)/1e3:.0f}K, "
          f"V={V}, n_bytes={args.n_bytes}", flush=True)


if __name__ == "__main__":
    main()
