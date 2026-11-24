"""Bare-minimum benchmark: ONLY the activation + its residual coupling.

Strips everything else away (attention, codebook, decoder, data pipeline):
pure pointwise stacks fitting a rich 1-D target. Four variants at matched
width/depth/budget:

  fip_vp     champion: gated VP-skip  out = cos(psi) h + sin(psi) phi(W2 phi(W1 h) * (W2 h))
             (gated variant, isometric (gamma, beta) init, learnable psi)
  fip_plain  plain residual, isometric activation:  out = h + phi(Wh)  (theta ~ U[-2,2])
  gelu_ln    the llama-MLP analog: pre-LN GELU 2-layer MLP residual
  gelu_plain GELU 2-layer MLP residual, no norm (the naive baseline)

Also records, per variant: activation RMS per depth block, full-stack
gradient norm, and test loss - the three health signals from the design
notes (variance doubling / phase aliasing, gradient flow, expressivity).

Run:  python experiments/bare_minimum.py [--depth 8 --width 128 ...]
"""
import argparse
import math
import sys
import os
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fipnet.activation import fip_wave, GAMMA, BETA
import experiments.block_v2 as bv


def target_fn(x):
    """Rich 1-D target: incommensurate sines + a cusp (broad spectrum)."""
    return (torch.sin(x) + 0.6 * torch.sin(3.1 * x + 0.5)
            + 0.4 * torch.sin(10.7 * x) + 0.25 * torch.sin(23.3 * x)
            + 0.2 * (x % 1.0 - 0.5) * 2.0) / 2.0


def make_data(n_train=4096, n_test=1024, x_max=10.0, in_dim=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    xtr = torch.rand(n_train, generator=g) * x_max
    xte = torch.linspace(0, x_max, n_test)
    ytr = target_fn(xtr).unsqueeze(-1)
    yte = target_fn(xte).unsqueeze(-1)
    xtr = xtr.unsqueeze(-1).repeat(1, in_dim)
    xte = xte.unsqueeze(-1).repeat(1, in_dim)
    return xtr, ytr, xte, yte


class FIPVPBlock(nn.Module):
    """The champion's gated VP-skip, verbatim from BlockV2('gated')."""
    def __init__(self, d, w1, w2, w3, psi):
        super().__init__()
        self.w1, self.w2, self.w3, self.psi = w1, w2, w3, psi
        self.psi_alpha0 = math.sin(psi.item()) if torch.is_tensor(psi) else None

    def forward(self, h):
        g = torch.einsum("bd,rd->br", h, self.w1)
        g = fip_wave(g)                                   # phi(W1 h)
        gate = torch.einsum("bd,rd->br", h, self.w2)      # W2 h (gating map)
        out = torch.einsum("br,dr->bd", g * gate, self.w3)
        c, s = math.cos(self.psi.item()), math.sin(self.psi.item())
        return c * h + s * fip_wave(out)                  # VP-skip

    def act_rms(self, h):
        return fip_wave(torch.einsum("bd,rd->br", h, self.w1)).pow(2).mean().sqrt()


class FIPPlainBlock(nn.Module):
    """Plain residual with the isometric activation."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.w.weight)
        self.theta = nn.Parameter(torch.empty(d).uniform_(-2.0, 2.0))

    def forward(self, h):
        z = torch.nn.functional.linear(h, self.w.weight) + self.theta
        return h + fip_wave(z, theta=self.theta)

    def act_rms(self, h):
        return fip_wave(self.w(h) + self.theta).pow(2).mean().sqrt()


class FIPHalfBlock(nn.Module):
    """Symmetrically normalized residual: (h + phi(Wh)) / sqrt(2).
    With orthogonal W and E[phi'^2]=1 this is an exact isometry in BOTH
    directions: Var(out) = (1+1)/2 = 1, E||J^T v||^2 = (1+1)/2 = 1."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.w.weight)
        self.theta = nn.Parameter(torch.empty(d).uniform_(-2.0, 2.0))

    def forward(self, h):
        z = torch.nn.functional.linear(h, self.w.weight) + self.theta
        return (h + fip_wave(z, theta=self.theta)) / math.sqrt(2.0)

    def act_rms(self, h):
        return fip_wave(self.w(h) + self.theta).pow(2).mean().sqrt()


class FIPRotBlock(nn.Module):
    """v2: complex rotation with wave-modulated phase (EURNN-style, FIP wave).

    State channels are paired (z[2k]=Re, z[2k+1]=Im). The block multiplies
    each complex pair by e^{i ang_k} where
        ang_k = theta_k + alpha_k * u_k,   u = fip_wave(W z)
    Exact isometry by construction: |z'| = |z| per pair, forward AND backward
    carry the rotation; nonlinearity enters as input-dependent PHASE
    (modulation), not as an additive branch - the Euler-formula algebra the
    VP-skip was missing.
    """
    def __init__(self, d, alpha0=0.1):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.w.weight)
        self.theta = nn.Parameter(torch.zeros(d // 2))
        self.alpha = nn.Parameter(torch.full((d // 2,), alpha0))
        # per-channel wave offset for the modulation source
        self.offs = nn.Parameter(torch.empty(d).uniform_(-2.0, 2.0))

    def forward(self, h):
        u = fip_wave(torch.nn.functional.linear(h, self.w.weight)
                     + self.offs)              # (N, W), bounded
        ang = self.theta + self.alpha * u[:, 0::2]      # (N, W/2)
        zr, zi = h[:, 0::2], h[:, 1::2]
        c, s = torch.cos(ang), torch.sin(ang)
        out = torch.empty_like(h)
        out[:, 0::2] = c * zr - s * zi
        out[:, 1::2] = s * zr + c * zi
        return out

    def act_rms(self, h):
        return u.abs().mean() * 0 + fip_wave(
            torch.nn.functional.linear(h, self.w.weight)
            + self.offs).pow(2).mean().sqrt()


class GELULNBlock(nn.Module):
    """llama-MLP analog: pre-LN 2-layer GELU residual."""
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, int(8 * d / 3), bias=False)
        self.w2 = nn.Linear(int(8 * d / 3), d, bias=False)
        self.act = nn.GELU()

    def forward(self, h):
        return h + self.w2(self.act(self.w1(self.ln(h))))

    def act_rms(self, h):
        return self.act(self.w1(self.ln(h))).pow(2).mean().sqrt()


class GELUPlainBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w1 = nn.Linear(d, int(8 * d / 3), bias=False)
        self.w2 = nn.Linear(int(8 * d / 3), d, bias=False)
        self.act = nn.GELU()

    def forward(self, h):
        return h + self.w2(self.act(self.w1(h)))

    def act_rms(self, h):
        return self.act(self.w1(h)).pow(2).mean().sqrt()


class Stack(nn.Module):
    def __init__(self, kind, depth, width, in_dim=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        self.enter = nn.Linear(in_dim, width, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            if kind == "fip_vp":
                d = width
                w1 = nn.Parameter(torch.empty(64, d))
                w2 = nn.Parameter(torch.empty(64, d))
                w3 = nn.Parameter(torch.empty(d, 64))
                for p in (w1, w2):
                    nn.init.orthogonal_(p)      # (r, D), orthonormal rows
                nn.init.orthogonal_(w3)
                psi = nn.Parameter(torch.full((1,), math.asin(0.7)))
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
        self.out = nn.Linear(width, 1, bias=False)

    def forward(self, x):
        h = self.enter(x)
        rms = []
        for b in self.blocks:
            rms.append(b.act_rms(h).detach())
            h = b(h)
        return self.out(h), rms


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_kind(kind, depth, width, steps, lr, xtr, ytr, xte, yte, device, seed=0):
    m = Stack(kind, depth, width, seed=seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = xtr.shape[0]
    bs = 1024
    hist = []
    gnorm_last = 0.0
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (bs,), device=device)
        pred, _ = m(xtr[idx])
        loss = (pred - ytr[idx]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm_last = torch.sqrt(sum(p.grad.norm() ** 2 for p in m.parameters()
                                    if p.grad is not None)).item()
        opt.step()
        if s % (steps // 10 or 1) == 0 or s == 1:
            with torch.no_grad():
                pte, _ = m(xte)
                te = (pte - yte).pow(2).mean().item()
            hist.append((s, loss.item(), te))
    with torch.no_grad():
        pte, rms = m(xte)
        te = (pte - yte).pow(2).mean().item()
        tr = (m(xtr[:4096])[0] - ytr).pow(2).mean().item()
    return dict(kind=kind, depth=depth, params=count_params(m), test=te,
                train=tr, gnorm=gnorm_last, time=time.time() - t0,
                curve=" ".join(f"{s}:{te * 1000:.2f}" for s, _, te in hist),
                rms=f"{rms[0].item():.3f}..{rms[-1].item():.3f}" if rms else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8,16,32")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = a.device
    xtr, ytr, xte, yte = make_data()
    xtr, ytr, xte, yte = [t.to(device) for t in (xtr, ytr, xte, yte)]
    # no-skill reference: predicting the mean
    floor = (yte - ytr.mean()).pow(2).mean().item()
    print(f"target variance (no-skill floor): {floor * 1000:.2f}e-3\n")
    kinds = ["fip_vp", "fip_plain", "gelu_ln", "gelu_plain"]
    for depth in [int(d) for d in a.depths.split(",")]:
        print(f"== depth {depth} ==")
        for kind in kinds:
            r = run_kind(kind, depth, a.width, a.steps, a.lr, xtr, ytr, xte, yte, device)
            print(f"  {kind:<10} params {r['params']:7d}  test {r['test']*1000:7.2f}e-3"
                  f"  train {r['train']*1000:7.2f}e-3  |g| {r['gnorm']:8.3f}"
                  f"  actRMS {r['rms']}  [{r['time']:.0f}s]")
            print(f"             curve(1e-3): {r['curve']}")
    print("\n(goal: test loss << floor; watch actRMS growth = variance doubling,")
    print(" |g| collapse = gradient vanishing)")


if __name__ == "__main__":
    main()
