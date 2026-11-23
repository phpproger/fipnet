"""Char-level language model experiment: FIP-Net vs classic embedding MLP.

Task: predict the next character from a fixed window of N=32 previous
characters (Bengio-style window LM) on TinyShakespeare (V=65).

FIP model: structured dyadic harmonic char codes (omega_j = 2^j / V with
quadrature T/4 pairs) lifted by one shared Linear(2*n_oct -> D), gathered
from a per-step V x D LUT — no embedding table, no V x D output matrix.
Training uses fused capturable Adam inside a CUDA graph (static shapes)
so the many tiny activation kernels don't pay launch overhead.

Usage: python experiments/char_lm.py [--steps 16000] [--window 32] [--no-graph]
"""

import argparse
import sys
import time
import urllib.request

import torch

sys.path.insert(0, ".")

from fipnet.model import FIPCharLM, MlpCharLM
from fipnet.graph_step import GraphedStep

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA = "./data/tinyshakespeare.txt"


def load_text(device):
    import os
    if not os.path.exists(DATA):
        os.makedirs("./data", exist_ok=True)
        print("downloading tinyshakespeare ...")
        urllib.request.urlretrieve(URL, DATA)
    with open(DATA, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    return ids, len(chars), chars


def run(name, model, ids, window, steps, batch, device, lr, use_graph):
    torch.manual_seed(0)
    model = model.to(device)
    n = ids.shape[0]
    train_ids, val_ids = ids[:int(0.9 * n)], ids[int(0.9 * n):]

    def sample(data, count):
        pos = torch.randint(0, data.shape[0] - window - 1, (count,), device=device)
        off = torch.arange(window, device=device)
        return (data[pos.unsqueeze(1) + off.unsqueeze(0)], data[pos + window])

    torch.manual_seed(1)
    vx, vy = sample(val_ids, 4096)
    torch.manual_seed(0)

    opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True, capturable=use_graph)
    stepper = None
    if use_graph:
        stepper = GraphedStep(model, opt, vx[:batch], vy[:batch])

    t0 = time.time()
    best_val_acc, n_steps = 0.0, 0
    for step in range(1, steps + 1):
        x, y = sample(train_ids, batch)
        if use_graph:
            loss = stepper.step(x, y)
        else:
            opt.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
            loss.backward()
            opt.step()
        n_steps += 1
        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                vacc = (model.predict(vx) == vy).float().mean().item()
            best_val_acc = max(best_val_acc, vacc)
            print(f"  [{name}] step {step}: val acc {vacc*100:.2f}%  "
                  f"({(time.time()-t0)/n_steps*1000:.2f} ms/step)", flush=True)
    dt = time.time() - t0
    return dict(name=name, acc=best_val_acc,
                params=sum(p.numel() for p in model.parameters()),
                seconds=dt, ms_per_step=dt / n_steps * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, V, chars = load_text(device)
    use_graph = (device == "cuda") and not args.no_graph
    print(f"device: {device}, vocab {V}, corpus {ids.shape[0]} chars, "
          f"window {args.window}, cuda-graph: {use_graph}")

    results = []
    results.append(run("fip-charlm", FIPCharLM(V, window=args.window, dim=args.dim,
                                               depth=6, code_dim=16),
                       ids, args.window, args.steps, args.batch, device,
                       args.lr, use_graph))
    results.append(run("mlp-charlm-baseline", MlpCharLM(V, window=args.window,
                                                        dim=args.dim, depth=6),
                       ids, args.window, args.steps, args.batch, device,
                       args.lr, use_graph))

    print("\n=== char-LM summary (best val next-char accuracy) ===")
    print(f"{'model':22s} {'acc %':>7s} {'params':>9s} {'ms/step':>8s} {'sec':>7s}")
    for r in results:
        print(f"{r['name']:22s} {r['acc']*100:7.2f} {r['params']:9d} "
              f"{r['ms_per_step']:8.2f} {r['seconds']:7.1f}")


if __name__ == "__main__":
    main()
