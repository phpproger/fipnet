"""Component ablation on TinyStories BPE-2048 (window LM, W=64, D=128).

Which substitution breaks at scale? Grid over:
  input:  embedding table  |  harmonic (dyadic+lift+LUT, no table)
  blocks: ReLU             |  parabolic phi    |  triangular phi
          (+ optional LayerNorm / residual on top of periodic blocks)
  head:   softmax Linear(D,V) + CE  |  low-rank phase codes + resonance CE

Usage: python experiments/ablation_tinystories.py [--steps 8000]
"""

import argparse
import sys
import time

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from fipnet.model import IsometricBlock, LowRankPhaseBottleneck, _uniform_phase_init
from fipnet.activation import fip_wave, fip_tri, GAMMA, BETA
from fipnet.graph_step import GraphedStep
from experiments.tiny_stories import load_chunks, DATA_DIR

V, D, DEPTH = 2048, 128, 6


class Block(nn.Module):
    """Linear + activation, optionally wrapped with LayerNorm / residual.

    residual_mode='scaled' uses the variance-preserving skip
    sqrt(1-a^2)*h + a*phi(...) instead of plain summation (plain residual
    grows pre-activation amplitude and causes phase aliasing in periodic
    stacks).
    """

    def __init__(self, dim, act, norm=False, residual=False, skip_alpha=None):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.act = act
        self.norm = norm
        self.residual = residual
        self.skip_alpha = skip_alpha
        if act != "relu":
            nn.init.orthogonal_(self.lin.weight)
        if norm:
            self.ln = nn.LayerNorm(dim)
        if act != "relu":
            self.bias = nn.Parameter(torch.zeros(dim))
            _uniform_phase_init(self.bias)
            self.lin.bias = None

    def phi(self, z):
        if self.act == "relu":
            return F.relu(z)
        if self.act == "parabolic":
            return fip_wave(z, GAMMA, BETA)
        return fip_tri(z)

    def forward(self, h):
        z = F.linear(h, self.lin.weight,
                     self.lin.bias if self.lin.bias is not None else self.bias)
        out = self.phi(z)
        if self.norm:
            out = self.ln(out)
        if self.residual:
            if self.skip_alpha is not None:
                a = self.skip_alpha
                return (1.0 - a * a) ** 0.5 * h + a * out
            return h + out
        return out


class AblationLM(nn.Module):
    def __init__(self, input_mode="embedding", act="relu", head="softmax",
                 norm=False, residual=False, skip_alpha=None, window=64, dim=D,
                 code_dim=32, n_oct=11, lr_scale=1.0, depth=DEPTH,
                 flat_init="orthogonal"):
        super().__init__()
        self.window = window
        self.head_mode = head
        self.input_mode = input_mode
        self.act = act
        if input_mode == "embedding":
            self.embed = nn.Embedding(V, dim)
        else:  # harmonic: dyadic quadrature code + shared lift (no table)
            self.n_oct = n_oct
            omega = (2.0 ** torch.arange(n_oct, dtype=torch.float32)) / V
            self.register_buffer("omega", omega)
            self.lift = nn.Linear(2 * n_oct, dim, bias=False)
            nn.init.orthogonal_(self.lift.weight)
            self.lift_bias = nn.Parameter(torch.zeros(dim))
            _uniform_phase_init(self.lift_bias)
        self.flat_proj = nn.Linear(window * dim, dim, bias=False)
        if flat_init == "orthogonal":
            nn.init.orthogonal_(self.flat_proj.weight)
            self.flat_bias = nn.Parameter(torch.zeros(dim))
            _uniform_phase_init(self.flat_bias)
        else:  # torch default Linear init
            nn.init.kaiming_uniform_(self.flat_proj.weight, a=math.sqrt(5))
            self.flat_bias = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(Block(dim, act, norm, residual, skip_alpha)
                                    for _ in range(depth))
        if head == "softmax":
            self.head = nn.Linear(dim, V)
        else:
            self.decoder = LowRankPhaseBottleneck(dim, V, code_dim, "fourier",
                                                  trainable_codes=True)

    def _phi(self, z):
        return (fip_wave(z, GAMMA, BETA) if self.act == "parabolic"
                else (F.relu(z) if self.act == "relu" else fip_tri(z)))

    def _in(self, tokens):
        if self.input_mode == "embedding":
            return self.embed(tokens)                       # (B, W, D)
        z = tokens.unsqueeze(-1).float() * self.omega       # (B, W, n_oct)
        code = torch.cat([fip_wave(z, GAMMA, BETA, 0.0),
                          fip_wave(z, GAMMA, BETA, 1.0)], dim=-1)
        h = F.linear(code, self.lift.weight, self.lift_bias)
        return fip_wave(h, GAMMA, BETA)

    def features(self, tokens):
        h = self._in(tokens)
        h = self._phi(F.linear(h.reshape(h.shape[0], -1),
                               self.flat_proj.weight, self.flat_bias))
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, x, y):
        h = self.features(x)
        if self.head_mode == "softmax":
            return F.cross_entropy(self.head(h), y)
        return self.decoder.loss(h, y, mode="ce")

    def predict(self, x):
        h = self.features(x)
        if self.head_mode == "softmax":
            return self.head(h).argmax(-1)
        return self.decoder.decode(h)


VARIANTS = {
    "A classic: emb+relu+softmax":      dict(input_mode="embedding", act="relu",      head="softmax"),
    "B emb+parabolic+softmax":          dict(input_mode="embedding", act="parabolic", head="softmax"),
    "C emb+triangular+softmax":         dict(input_mode="embedding", act="triangular",head="softmax"),
    "D emb+relu+phase-head(t):":        dict(input_mode="embedding", act="relu",      head="phase"),
    "E emb+parabolic+phase(t)":         dict(input_mode="embedding", act="parabolic", head="phase"),
    "F harmonic+parabolic+softmax":     dict(input_mode="harmonic",  act="parabolic", head="softmax"),
    "G full-fip (harm+phase t)":        dict(input_mode="harmonic",  act="parabolic", head="phase"),
    "G+ln: full-fip + LayerNorm":       dict(input_mode="harmonic",  act="parabolic", head="phase", norm=True),
    "G+res: full-fip + residual":       dict(input_mode="harmonic",  act="parabolic", head="phase", residual=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--variants", type=str, default=None)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    train, val = load_chunks(device, args.train_tokens, 1_000_000)
    W, B = args.window, args.batch

    def sample(data, count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        return data[pos.unsqueeze(1) + off.unsqueeze(0)].long(), data[pos + W].long()

    torch.manual_seed(1)
    vx, vy = sample(val, 4096)
    torch.manual_seed(0)

    names = args.variants.split(";") if args.variants else list(VARIANTS)
    print(f"{'variant':34s} {'val acc %':>9s} {'val loss':>9s} {'sec':>6s}")
    for name in names:
        cfg = VARIANTS[name]
        torch.manual_seed(0)
        model = AblationLM(window=W, **cfg).to(device)
        if cfg["head"] == "phase":
            model.decoder.codes()          # materialize trainable codebook param
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=3e-3 if cfg["act"] == "relu" else 3e-4,
                               fused=True, capturable=True)
        gs = GraphedStep(model, opt, vx[:B], vy[:B])
        t0 = time.time()
        best_acc, best_loss = 0.0, 9e9
        for step in range(1, args.steps + 1):
            x, y = sample(train, B)
            loss = gs.step(x, y)
            if step % 2000 == 0 or step == args.steps:
                with torch.no_grad():
                    vacc = (model.predict(vx) == vy).float().mean().item()
                    vloss = model.loss(vx, vy).item()
                best_acc, best_loss = max(best_acc, vacc), min(best_loss, vloss)
                print(f"  [{name[:30]:30s}] step {step}: acc {vacc*100:6.2f}% loss {vloss:.3f} "
                      f"(train {loss.item():.3f})", flush=True)
        print(f"{name:34s} {best_acc*100:9.2f} {best_loss:9.3f} {time.time()-t0:6.0f}  params {params/1e6:.2f}M", flush=True)


if __name__ == "__main__":
    main()
