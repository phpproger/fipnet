"""Block-level modifications to break the 25.8% ceiling (champion H1+skip).

P1  phase-lr x3   : separate param group with 3x lr for all phase/bias
                    parameters (theta vectors) — curvature placement learns
                    faster than the linear maps.
P2  channel-beta  : learnable per-channel frequency beta_i (init sqrt(2/5))
                    and learnable skip angle psi_i (init sin psi = 0.7):
                    z = beta .* (W h) + theta,  out = cos(psi) h + sin(psi) phi(z).
                    Isometry holds at init; channels then specialize into
                    linear carriers / quadratic detectors / high-freq filters.
P3  gated block   : W3( phi(W1 h) * (W2 h) ) with r = D/3 so that
                    3 * D * r = D^2 (same weights as the plain block);
                    multiplicative mixing creates the full frequency spectrum.
P4  equal-params  : width D chosen so total params == A classic (1.67M),
                    while keeping 0 vocabulary parameters.

Usage: python experiments/block_v2.py [--steps 8000] [--dims ...]
"""

import argparse
import math

SQRT2 = math.sqrt(2.0)
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from fipnet.model import LowRankPhaseBottleneck, _uniform_phase_init
from fipnet.activation import fip_wave, fip_wave_c, GAMMA, BETA
from fipnet.graph_step import GraphedStep
from experiments.tiny_stories import load_chunks

V = 2048
TK = r"C:\Users\ignat\PycharmProjects\glm\data\tinystories_tokenizer.json"


class BlockV2(nn.Module):
    def __init__(self, dim, kind="plain", skip_alpha=0.7):
        super().__init__()
        self.kind = kind
        self.psi_alpha0 = skip_alpha
        if kind != "gated":                 # gated carries no dense W (bug fix:
                                            # it previously held a dead D^2
                                            # weight unused by forward)
            self.weight = nn.Parameter(torch.empty(dim, dim))
            nn.init.orthogonal_(self.weight)
            self.bias = nn.Parameter(torch.zeros(dim))
            _uniform_phase_init(self.bias)
        if kind == "channel":
            self.beta = nn.Parameter(torch.full((dim,), BETA))
            self.psi = nn.Parameter(torch.full((dim,), math.asin(skip_alpha)))
        elif kind == "gated":
            r = max(1, dim // 3)
            # F.linear convention: weight is (out, in); 3*D*r = D^2 params
            self.w1 = nn.Parameter(torch.empty(r, dim))
            self.w2 = nn.Parameter(torch.empty(r, dim))
            self.w3 = nn.Parameter(torch.empty(dim, r))
            nn.init.orthogonal_(self.w1)   # (r, dim): orthonormal rows
            nn.init.orthogonal_(self.w2)
            nn.init.orthogonal_(self.w3)   # (dim, r): orthonormal columns
            self.psi = nn.Parameter(torch.full((dim,), math.asin(skip_alpha)))
        else:
            self.skip_alpha = skip_alpha

    def iso_prior(self):
        """Soft Isometry Prior: squared deviation of the learnable per-channel
        parameters from their isometric init values. Added with a small
        coefficient (lambda_iso ~ 1e-3) to the training loss, it keeps the
        stack variance-preserving at depth while letting channels
        specialize locally (linear carriers / quadratic detectors)."""
        if self.kind == "channel":
            return ((self.beta - BETA) ** 2).mean() \
                 + ((torch.sin(self.psi) - self.psi_alpha0) ** 2).mean()
        if self.kind == "gated":
            return ((torch.sin(self.psi) - self.psi_alpha0) ** 2).mean()
        # half: nothing to anchor - exact isometry by construction
        return self.weight.new_zeros(())

    def forward(self, h):
        if self.kind == "half":
            z = F.linear(h, self.weight) + self.bias
            return (h + fip_wave_c(z, GAMMA, BETA)) / SQRT2
        if self.kind == "gated":
            out = F.linear(fip_wave_c(F.linear(h, self.w1), GAMMA, BETA)
                           * F.linear(h, self.w2), self.w3)
            return torch.cos(self.psi) * h + torch.sin(self.psi) * out
        z = F.linear(h, self.weight)
        if self.kind == "channel":
            out = fip_wave_c(self.beta * z + self.bias, GAMMA, BETA)
            return torch.cos(self.psi) * h + torch.sin(self.psi) * out
        out = fip_wave_c(z + self.bias, GAMMA, BETA)
        a = self.skip_alpha
        return (1.0 - a * a) ** 0.5 * h + a * out


class FIPV2(nn.Module):
    """Byte-tuple harmonic input + BlockV2 stack + trainable phase head."""

    def __init__(self, window=64, dim=128, depth=6, code_dim=32, n_bytes=8,
                 d_code=32, block_kind="plain", skip_alpha=0.7, flat_init="orthogonal"):
        super().__init__()
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(TK)
        import numpy as np
        rows = np.zeros((V, n_bytes), dtype=np.int64)
        for i in range(V):
            s = tk.id_to_token(i).replace("\u0120", " ").replace("\u010a", "\n")
            b = s.encode("utf-8")[:n_bytes]
            rows[i, :len(b)] = list(b)
        self.register_buffer("bytes_mat", torch.from_numpy(rows))
        self.n_bytes, self.d_code = n_bytes, d_code
        self.omega = nn.Parameter(torch.randn(n_bytes, d_code) * 0.5)
        self.theta = nn.Parameter(torch.zeros(n_bytes, d_code))
        _uniform_phase_init(self.theta)
        self.lift = nn.Linear(d_code, dim, bias=False)
        nn.init.orthogonal_(self.lift.weight)
        self.lift_bias = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.lift_bias)
        self.flat_proj = nn.Linear(window * dim, dim, bias=False)
        if flat_init == "orthogonal":
            nn.init.orthogonal_(self.flat_proj.weight)
            self.flat_bias = nn.Parameter(torch.zeros(dim))
            _uniform_phase_init(self.flat_bias)
        else:
            nn.init.kaiming_uniform_(self.flat_proj.weight, a=math.sqrt(5))
            self.flat_bias = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(BlockV2(dim, block_kind, skip_alpha)
                                    for _ in range(depth))
        self.decoder = LowRankPhaseBottleneck(dim, V, code_dim, "fourier",
                                              trainable_codes=True)

    def _lut(self):
        z = self.bytes_mat.float().unsqueeze(-1) * self.omega.unsqueeze(0) \
            + self.theta.unsqueeze(0)
        return fip_wave_c(z, GAMMA, BETA).sum(1)

    def features(self, tokens):
        h = self._lut()[tokens]
        h = fip_wave_c(F.linear(h, self.lift.weight, self.lift_bias), GAMMA, BETA)
        h = h.reshape(h.shape[0], -1)
        h = fip_wave_c(F.linear(h, self.flat_proj.weight, self.flat_bias), GAMMA, BETA)
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, x, y):
        return self.decoder.loss(self.features(x), y, mode="ce")

    def predict(self, x):
        return self.decoder.decode(self.features(x))


def pick_dim(target_params, window, depth, code_dim):
    for dim in range(120, 260, 2):
        n = (window * dim * dim            # flat_proj
             + depth * dim * dim           # plain blocks
             + depth * dim * 4             # block biases ~
             + dim * dim * 0               # (lift small)
             + code_dim * dim              # head
             + V * code_dim                # codebook
             + 3_000)
        if n >= target_params:
            return dim
    return 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--jobs", type=str, default="P1,P2,P3,P4")
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

    eq_dim = pick_dim(1_670_000, W, 6, 32)
    print(f"equal-params width: D={eq_dim}")

    jobs = {
        "P1": ("P1 phase-lr x3 (plain skip)", dict(block_kind="plain"), 3e-4, True, (0.8, 0.999)),
        "P2": ("P2 channel-beta + learnable skip", dict(block_kind="channel"), 3e-4, False, (0.8, 0.999)),
        "P3": ("P3 gated r=D/3 + skip", dict(block_kind="gated"), 3e-4, False, (0.8, 0.999)),
        "P4": (f"P4 equal-params D={eq_dim} plain", dict(block_kind="plain", dim=eq_dim), 3e-4, False, (0.8, 0.999)),
    }
    for key in args.jobs.split(","):
        name, kw, lr, phase_lr_boost, betas = jobs[key]
        torch.manual_seed(0)
        model = FIPV2(window=W, **kw).to(device)
        model.decoder.codes()
        if phase_lr_boost:
            phase_params, rest = [], []
            for n_, p in model.named_parameters():
                (phase_params if ("theta" in n_ or "bias" in n_) else rest).append(p)
            opt = torch.optim.Adam([
                {"params": rest, "lr": lr},
                {"params": phase_params, "lr": 3 * lr},
            ], betas=betas, fused=True, capturable=True)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas,
                                   fused=True, capturable=True)
        gs = GraphedStep(model, opt, vx[:B], vy[:B])
        t0 = time.time()
        best = (0.0, 9e9)
        traj = []
        for step in range(1, args.steps + 1):
            x, y = sample(train, B)
            loss = gs.step(x, y)
            if step % 2000 == 0 or step == args.steps:
                with torch.no_grad():
                    vacc = (model.predict(vx) == vy).float().mean().item()
                    vloss = model.loss(vx, vy).item()
                best = (max(best[0], vacc), min(best[1], vloss))
                traj.append(f"{step//1000}k:{vacc*100:.1f}")
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:36s} BEST {best[0]*100:6.2f}% loss {best[1]:.3f} "
              f"params {params/1e6:.2f}M [{' '.join(traj)}] [{time.time()-t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
