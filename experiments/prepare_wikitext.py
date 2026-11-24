"""Prepare wikitext-103 for FIP-Net: raw download -> BPE -> token shards.

Deliberately dependency-light: NO datasets/pandas/pyarrow (their numpy ABI
chains break across Kaggle images). Just urllib + zipfile + tokenizers.

    python experiments/prepare_wikitext.py --vocab 8192 --n-bytes 12 \
        --max-tokens 100000000 --out-dir data

Outputs: train_tokens.npy / val_tokens.npy (uint16), wikitext_tokenizer.json,
bytes_mat.pt (V x n_bytes rows).
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

URLS = [
    "https://wikitext.smerity.com/wikitext-103-raw-v1.zip",
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
]


def fetch_raw(out_dir):
    zip_path = os.path.join(out_dir, "wikitext-103-raw-v1.zip")
    for url in URLS:
        try:
            print(f"downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                names = [n for n in z.namelist() if n.endswith(".raw")]
                for n in names:
                    z.extract(n, out_dir)
            return names
        except Exception as e:                                   # noqa: BLE001
            print(f"  failed: {e}", flush=True)
    raise SystemExit("all wikitext mirrors failed - download manually to "
                     + out_dir)


def find_split(extracted, key):
    for path in extracted:
        if key in path and (path.endswith(key) or key + "." in path
                            or path.split("/")[-1] == key):
            return os.path.join(*path.split("/")) if "/" in path else path
    # fallback: any path containing the key
    for path in extracted:
        if key in path:
            return path
    raise SystemExit(f"split {key} not found in {extracted}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n-bytes", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=100_000_000)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    names = fetch_raw(args.out_dir)
    extracted = [os.path.join(args.out_dir, n) for n in names]
    train_txt = find_split(extracted, "wiki.train.raw")
    test_txt = find_split(extracted, "wiki.test.raw")
    print(f"train file: {train_txt} ({os.path.getsize(train_txt)/1e6:.0f} MB)",
          flush=True)

    def lines(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                t = line.rstrip("\n")
                if t.strip():
                    yield t

    tok_path = os.path.join(args.out_dir, "wikitext_tokenizer.json")
    if not os.path.exists(tok_path):
        print("training BPE tokenizer ...", flush=True)
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab, special_tokens=["<|pad|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(lines(train_txt), trainer)
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

    def encode(path, cap):
        ids = []
        for line in lines(path):
            ids.extend(tok.encode(line).ids)
            if len(ids) >= cap:
                break
        return np.asarray(ids[:cap], dtype=np.uint16)

    print("tokenizing train ...", flush=True)
    tr = encode(train_txt, args.max_tokens)
    print("tokenizing test ...", flush=True)
    va = encode(test_txt, args.val_tokens)
    np.save(os.path.join(args.out_dir, "train_tokens.npy"), tr)
    np.save(os.path.join(args.out_dir, "val_tokens.npy"), va)
    print(f"done: train {len(tr)/1e6:.1f}M tokens, val {len(va)/1e3:.0f}K, "
          f"V={V}, n_bytes={args.n_bytes}", flush=True)


if __name__ == "__main__":
    main()
