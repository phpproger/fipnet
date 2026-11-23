"""Input fixes for FIP window LM on TinyStories BPE-2048 (W=64, D=128).

Implements the two table-free/tied input designs plus two optimizer/skip
hypotheses from the ablation analysis:
  H1 byte-tuple harmonics : code a token by the actual bytes of its string
      (up to M=8, zero-padded): sum_m phi(b_m * omega_m + theta_m), then a
      shared lift. Zero vocabulary parameters at all.
  H2 tied low-rank codebook : the same trainable V x K codebook used by the
      phase head is read on the input: h0 = phi(W_in c_v + theta).
      Single semantic bottleneck V x K on both sides.
  B-b1 : Adam beta1 = 0.8 for the parabolic stack (first-moment cancellation
      at phi' sign flips is reduced).
  G+skip : variance-preserving skip sqrt(1-a^2) h + a phi(...), a = 0.7.

Usage: python experiments/input_fixes.py [--steps 8000]
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from fipnet.model import LowRankPhaseBottleneck, _uniform_phase_init
from fipnet.activation import fip_wave, GAMMA, BETA
from fipnet.graph_step import GraphedStep
from experiments.ablation_tinystories import AblationLM, Block
from experiments.tiny_stories import load_chunks

V, D = 2048, 128


class ByteHarmonicLM(nn.Module):
    """Full-FIP with byte-tuple harmonic input: no vocabulary parameters."""

    def __init__(self, tokenizer_path, window=64, dim=D, depth=6, code_dim=32,
                 n_bytes=8, d_code=32, skip_alpha=None):
        super().__init__()
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tokenizer_path)
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
        nn.init.orthogonal_(self.flat_proj.weight)
        self.flat_bias = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.flat_bias)
        res = skip_alpha is not None
        self.blocks = nn.ModuleList(Block(dim, "parabolic", residual=res,
                                          skip_alpha=skip_alpha)
                                    for _ in range(depth))
        self.decoder = LowRankPhaseBottleneck(dim, V, code_dim, "fourier",
                                              trainable_codes=True)

    def _lut(self):
        # (V, M, dc) = phi(b_m * omega_m + theta_m), summed over byte positions
        z = self.bytes_mat.float().unsqueeze(-1) * self.omega.unsqueeze(0) \
            + self.theta.unsqueeze(0)
        return fip_wave(z, GAMMA, BETA).sum(1)              # (V, dc)

    def features(self, tokens):
        h = self._lut()[tokens]                              # (B, W, dc)
        h = fip_wave(F.linear(h, self.lift.weight, self.lift_bias), GAMMA, BETA)
        h = h.reshape(h.shape[0], -1)
        h = fip_wave(F.linear(h, self.flat_proj.weight, self.flat_bias), GAMMA, BETA)
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, x, y):
        return self.decoder.loss(self.features(x), y, mode="ce")

    def predict(self, x):
        return self.decoder.decode(self.features(x))


class TiedCodebookLM(nn.Module):
    """Full-FIP with the head's trainable V x K codebook reused as input:
    h0 = phi(W_in c_v + theta). One semantic bottleneck V x K on both sides."""

    def __init__(self, window=64, dim=D, depth=6, code_dim=32, skip_scale=None):
        super().__init__()
        self.decoder = LowRankPhaseBottleneck(dim, V, code_dim, "fourier",
                                              trainable_codes=True)
        self.lift = nn.Linear(code_dim, dim, bias=False)
        nn.init.orthogonal_(self.lift.weight)
        self.lift_bias = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.lift_bias)
        self.flat_proj = nn.Linear(window * dim, dim, bias=False)
        nn.init.orthogonal_(self.flat_proj.weight)
        self.flat_bias = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.flat_bias)
        self.blocks = nn.ModuleList(Block(dim, "parabolic")
                                    for _ in range(depth))

    def features(self, tokens):
        h = self.decoder.codes()[tokens]                     # (B, W, K), tied
        h = fip_wave(F.linear(h, self.lift.weight, self.lift_bias), GAMMA, BETA)
        h = h.reshape(h.shape[0], -1)
        h = fip_wave(F.linear(h, self.flat_proj.weight, self.flat_bias), GAMMA, BETA)
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, x, y):
        return self.decoder.loss(self.features(x), y, mode="ce")

    def predict(self, x):
        return self.decoder.decode(self.features(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    train, val = load_chunks(device, 20_000_000, 1_000_000)
    W, B = args.window, args.batch

    def sample(data, count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        return data[pos.unsqueeze(1) + off.unsqueeze(0)].long(), data[pos + W].long()

    torch.manual_seed(1)
    vx, vy = sample(val, 4096)
    torch.manual_seed(0)
    tk_path = r"C:\Users\ignat\PycharmProjects\glm\data\tinystories_tokenizer.json"

    jobs = [
        ("H1 byte-harmonic full-FIP", lambda: ByteHarmonicLM(tk_path, W), 3e-4, {}),
        ("H2 tied-codebook full-FIP", lambda: TiedCodebookLM(W), 3e-4, {}),
        ("B-b1: emb+parab+softmax b1=.8", lambda: AblationLM(input_mode="embedding", act="parabolic", head="softmax", window=W), 3e-4, {"betas": (0.8, 0.999)}),
        ("G+skip a=0.7 full-FIP", lambda: AblationLM(input_mode="harmonic", act="parabolic", head="phase", window=W), 3e-4, {"skip": True}),
    ]
    for name, build, lr, kw in jobs:
        torch.manual_seed(0)
        model = build().to(device)
        if hasattr(model, "decoder"):
            model.decoder.codes()          # materialize trainable codebook param
        if kw.get("skip"):
            for blk in model.blocks:       # variance-preserving skip a=0.7
                blk.residual = True
                blk.skip_alpha = 0.7
        opt_kw = {"fused": True, "capturable": True}
        if "betas" in kw:
            opt_kw["betas"] = kw["betas"]
        opt = torch.optim.Adam(model.parameters(), lr=lr, **opt_kw)
        gs = GraphedStep(model, opt, vx[:B], vy[:B])
        t0 = time.time()
        best = (0.0, 9e9)
        for step in range(1, args.steps + 1):
            x, y = sample(train, B)
            loss = gs.step(x, y)
            if step % 2000 == 0 or step == args.steps:
                with torch.no_grad():
                    vacc = (model.predict(vx) == vy).float().mean().item()
                    vloss = model.loss(vx, vy).item()
                best = (max(best[0], vacc), min(best[1], vloss))
                print(f"  [{name}] step {step}: acc {vacc*100:6.2f}% loss {vloss:.3f} "
                      f"(train {loss.item():.3f})", flush=True)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:34s} BEST acc {best[0]*100:6.2f} loss {best[1]:.3f} "
              f"params {params/1e6:.2f}M [{time.time()-t0:.0f}s]\n", flush=True)


if __name__ == "__main__":
    main()
