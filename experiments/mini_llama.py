"""MiniLlama baseline: the full modern transformer stack (llama2.c style) on the
same TinyStories window-LM task, for an honest head-to-head with ATT1.

Components: RMSNorm (pre-norm), RoPE, multi-head causal attention, SwiGLU MLP,
no biases, untied (or tied) embeddings, softmax CE head. This is the stack ATT1
deliberately avoids (norms + residual + embedding table + softmax head).

Usage:
  python experiments/mini_llama.py --steps 30000            # llama-equal (1.7M)
  python experiments/mini_llama.py --config small           # llama-iso (0.29M)
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from experiments.tiny_stories import load_chunks

V = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(window, head_dim, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(window, device=device).float()
    freqs = torch.outer(t, inv)                      # (W, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):                          # x: (B, H, W, hd)
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    c = cos[None, None]                               # (1, 1, W, hd/2)
    s = sin[None, None]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class LlamaAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        cos, sin = rope_cache(window, self.hd, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("mask", torch.tril(torch.ones(window, window, dtype=torch.bool)))

    def forward(self, x):
        B, W, D = x.shape
        q = self.q(x).view(B, W, self.heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, W, self.heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, W, self.heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        att = att.masked_fill(~self.mask, float("-inf"))
        ctx = torch.softmax(att, dim=-1) @ v
        return self.o(ctx.transpose(1, 2).reshape(B, W, D))


class LlamaBlock(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        hidden = int(8 * dim / 3)
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = LlamaAttention(dim, heads, window)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.w2(F.silu(self.w1(self.n2(x))) * self.w3(self.n2(x)))

    def attn_residual(self, x):
        return x + self.attn(self.n1(x))

    def ffn_residual(self, x):
        return x + self.w2(F.silu(self.w1(self.n2(x))) * self.w3(self.n2(x)))


class MiniLlama(nn.Module):
    def __init__(self, dim=160, layers=5, heads=4, window=64, tie=False, allpos=False,
                 vocab=None):
        super().__init__()
        self.window = window
        self.allpos = allpos
        self.embed = nn.Embedding(vocab or V, dim)
        self.blocks = nn.ModuleList(LlamaBlock(dim, heads, window) for _ in range(layers))
        self.norm_f = RMSNorm(dim)
        if tie:
            self.head = None
        else:
            self.head = nn.Linear(dim, vocab or V, bias=False)
        self.head_w = None

    def features(self, tokens, full=False):
        h = self.embed(tokens)
        for b in self.blocks:
            h = b(h)
        if full:
            return self.norm_f(h)                   # (B, W, D) all positions
        g = h[:, -1]
        return self.norm_f(g)

    def _logits(self, h):
        if self.head is None:
            return F.linear(h, self.embed.weight)
        return self.head(h)

    def loss(self, x, y):
        h = self.features(x, full=self.allpos)
        if self.allpos:
            logits = self._logits(h)                # (B, W, V)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        return F.cross_entropy(self._logits(h), y)

    def predict(self, x):
        h = self.features(x, full=self.allpos)
        logits = self._logits(h)
        if self.allpos:
            return logits.argmax(-1)                # (B, W)
        return logits.argmax(-1)


CONFIGS = {
    # ~1.7M: iso with classic A (embedding+ReLU+softmax flat)
    "equal": dict(dim=160, layers=5, heads=4),
    # ~0.29M: iso with ATT1 champion (tied head to keep the budget honest)
    "small": dict(dim=64, layers=6, heads=4, tie=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="equal", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bench", action="store_true", help="speed/VRAM only, no training")
    ap.add_argument("--ckpt", default=None, help="checkpoint file; saves every 2k steps")
    ap.add_argument("--resume", action="store_true", help="resume from --ckpt if it exists")
    ap.add_argument("--allpos", action="store_true",
                    help="train CE over all W positions (standard LM protocol)")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    train, val = load_chunks(device, args.train_tokens, 1_000_000)
    W, B = args.window, args.batch

    def sample(data, count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        x = data[idx].long()
        if args.allpos:
            return x, data[idx + 1].long()      # shifted targets, all positions
        return x, data[pos + W].long()

    torch.manual_seed(1)
    vx, vy = sample(val, 4096)
    torch.manual_seed(0)
    m = MiniLlama(window=W, allpos=args.allpos, **CONFIGS[args.config]).to(device)
    params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, betas=(0.9, 0.95), fused=True, capturable=True)

    start_step, best, traj = 1, (0.0, 9e9), []
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        st = torch.load(args.ckpt, map_location=device)
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step, best, traj = st["step"] + 1, st["best"], st["traj"]
        print(f"resumed at step {st['step']}", flush=True)

    # warmup + bench
    for _ in range(20):
        x, y = sample(train, B)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m._logits(m.features(x)), y).backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n = 100
    for _ in range(n):
        x, y = sample(train, B)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m._logits(m.features(x)), y).backward()
        opt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / n
    vram = torch.cuda.max_memory_allocated() / 2**20
    print(f"llama[{args.config}] {params/1e6:.2f}M params: {ms:.2f} ms/step (eager), "
          f"peak VRAM {vram:.0f} MiB", flush=True)
    if args.bench:
        return

    best = (0.0, 9e9)
    traj = []
    for s in range(start_step, args.steps + 1):
        x, y = sample(train, B)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m._logits(m.features(x)), y)
        loss.backward()
        opt.step()
        if s % 5000 == 0:
            with torch.no_grad():
                vacc = (m.predict(vx) == vy).float().mean().item()
                vloss = m.loss(vx, vy).item()
            best = (max(best[0], vacc), min(best[1], vloss))
            traj.append(f"{s//1000}k:{vacc*100:.1f}")
            torch.cuda.empty_cache()   # eval logits (~1GB) must not sit in cache
        if args.ckpt and s % 2000 == 0:
            tmp = args.ckpt + ".tmp"          # atomic: reboot/kill-safe
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                        "step": s, "best": best, "traj": traj}, tmp)
            os.replace(tmp, args.ckpt)
    print(f"MiniLlama[{args.config}] 30k: BEST {best[0]*100:6.2f}% loss {best[1]:.3f} "
          f"params {params/1e6:.2f}M [{' '.join(traj)}]", flush=True)


if __name__ == "__main__":
    main()
