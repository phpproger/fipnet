"""Bare-minimum on REAL text: activation + residual coupling only.

Next-token prediction on TinyStories (BPE-2048), window 64, last-position
loss. Stripped to the core question from the synthetic probe (which showed
the champion's gated VP-skip DEGRADING with depth while plain-residual
isometric stacks crush it):

  - input: FROZEN random projection (no learned embedding table)
  - head: trained linear width->V (identical everywhere)
  - stack variants: fip_vp / fip_plain / gelu_ln / gelu_plain (see
    bare_minimum.py)
  - matched depth/width/steps/lr; params reported (depths can be re-picked
    for param-matched comparisons)

Run: python experiments/bare_text.py --depths 8,16 --steps 3000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bare_minimum import (FIPVPBlock, FIPPlainBlock, FIPHalfBlock,
                                      FIPRotBlock, GELULNBlock, GELUPlainBlock)


class TextStack(nn.Module):
    def __init__(self, kind, depth, width, vocab, window, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.embed = nn.Embedding(vocab, width).requires_grad_(False)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.randn(vocab, width, generator=g) / (width ** 0.5))
        self.kind = kind
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            if kind == "fip_vp":
                d = width
                w1 = nn.Parameter(torch.empty(64, d))
                w2 = nn.Parameter(torch.empty(64, d))
                w3 = nn.Parameter(torch.empty(d, 64))
                for p in (w1, w2):
                    nn.init.orthogonal_(p)
                nn.init.orthogonal_(w3)
                psi = nn.Parameter(torch.full((1,), torch.asin(torch.tensor(0.7)).item()))
                self.blocks.append(FIPVPBlock(d, w1, w2, w3, psi))
            elif kind == "fip_plain":
                self.blocks.append(FIPPlainBlock(width))
            elif kind == "fip_half":
                self.blocks.append(FIPHalfBlock(width))
            elif kind == "fip_rot":
                self.blocks.append(FIPRotBlock(width))
            elif kind == "gelu_ln":
                self.blocks.append(GELULNBlock(width))
            else:
                self.blocks.append(GELUPlainBlock(width))
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, x):
        B, W = x.shape
        h = self.embed(x).reshape(B * W, -1)       # blocks are pointwise (N, D)
        for b in self.blocks:
            h = b(h)
        return self.head(h).reshape(B, W, -1)      # (B, W, V)


def loss_last(logits, y):
    return nn.functional.cross_entropy(logits[:, -1], y[:, -1])


def acc_last(logits, y):
    return (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item()


def run_kind(kind, depth, width, steps, lr, batch, window, vocab, train, vx, vy,
             device, seed=0):
    torch.manual_seed(seed)
    m = TextStack(kind, depth, width, vocab, window, seed=seed).to(device)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=lr)
    V = m.head.out_features
    hist, gnorm = [], 0.0
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = torch.randint(0, train.shape[0] - window - 1, (batch,), device=device)
        off = torch.arange(window, device=device)
        x = train[idx.unsqueeze(1) + off.unsqueeze(0)].long()
        y = train[idx.unsqueeze(1) + off.unsqueeze(0) + 1].long()
        logits = m(x)
        loss = loss_last(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.sqrt(sum(p.grad.norm() ** 2
                               for p in m.parameters() if p.grad is not None)).item()
        opt.step()
        if s % (steps // 5 or 1) == 0 or s == 1:
            with torch.no_grad():
                va = acc_last(m(vx), vy)
            hist.append((s, va))
            torch.cuda.empty_cache() if device.type == "cuda" else None
    with torch.no_grad():
        va = acc_last(m(vx), vy)
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return dict(kind=kind, depth=depth, params=n_params, acc=va,
                gnorm=gnorm, time=time.time() - t0,
                curve=" ".join(f"{s}:{a*100:.1f}" for s, a in hist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="8,16")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--data-dir",
                    default=r"C:/Users/ignat/PycharmProjects/future/tinystories_full_bpe2048")
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    tr = np.load(os.path.join(a.data_dir, "train2048.npy"), mmap_mode="r")
    va = np.load(os.path.join(a.data_dir, "val2048.npy"), mmap_mode="r")
    train = torch.from_numpy(np.array(tr[: a.train_tokens])).to(device).to(torch.int32)
    vocab = int(np.array(va[:100_000]).max()) + 1
    g = torch.Generator(device="cpu").manual_seed(1)
    W = a.window
    pos = torch.randint(0, 500_000 - W - 1, (2048,), generator=g).to(device)
    off = torch.arange(W, device=device)
    val_t = torch.from_numpy(np.array(va[:500_000])).to(device).to(torch.int32)
    vx = val_t[pos.unsqueeze(1) + off.unsqueeze(0)].long()
    vy = val_t[pos.unsqueeze(1) + off.unsqueeze(0) + 1].long()
    floor = 100.0 / vocab  # unigram-noise sanity (acc, %)
    print(f"vocab {vocab}  window {W}  steps {a.steps}  "
          f"chance-level acc ~ {1.0/vocab*100:.2f}%\n")
    for depth in [int(d) for d in a.depths.split(",")]:
        print(f"== depth {depth} ==")
        for kind in ["fip_vp", "fip_plain", "fip_half", "fip_rot", "gelu_ln", "gelu_plain"]:
            r = run_kind(kind, depth, a.width, a.steps, a.lr, a.batch, W,
                         vocab, train, vx, vy, device)
            print(f"  {kind:<10} params {r['params']:7d}  acc {r['acc']*100:6.2f}%"
                  f"  |g| {r['gnorm']:8.3f}  [{r['time']:.0f}s]")
            print(f"             curve: {r['curve']}")


if __name__ == "__main__":
    main()
