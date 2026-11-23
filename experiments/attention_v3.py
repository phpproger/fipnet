"""Attention-based context mixing for FIP window LM (TinyStories BPE-2048).

Replaces the flat projection (W*D -> D, ~1M params, init-sensitive) with
causal self-attention over the window:
  - softmax attention rows are convex combinations of bounded phi-values ->
    output stays bounded with NO normalization (fits the FIP design);
  - harmonic positional encoding (no table): pe = phi(pos * omega_p + theta_p);
  - last causal position carries the whole context -> blocks -> phase head.

ATT1: byte-harmonic input + attention + channel-beta blocks + phase head
ATT2: embedding input + attention + ReLU blocks + softmax head (reference)

Usage: python experiments/attention_v3.py [--steps 8000]
"""

import argparse
import math
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
from experiments.block_v2 import BlockV2
from experiments.tiny_stories import load_chunks

V = 2048
TK = r"C:\Users\ignat\PycharmProjects\glm\data\tinystories_tokenizer.json"


class CausalAttention(nn.Module):
    def __init__(self, dim, window, heads=1, pos_mode="harmonic"):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.pos_mode = pos_mode
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(window, window, dtype=torch.bool)))
        if pos_mode == "rope":
            # cos/sin computed ONCE at init (cache), not in the training loop:
            # the SFU-free constraint applies to the activation, not to a
            # precomputed constant table
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.dh, 2, dtype=torch.float32) / self.dh))
            t = torch.arange(window, dtype=torch.float32)
            freqs = torch.outer(t, inv)                      # (W, dh/2)
            self.register_buffer("rope_cos", torch.cos(freqs), persistent=False)
            self.register_buffer("rope_sin", torch.sin(freqs), persistent=False)

    def _rope(self, x):                                      # (B, H, W, dh)
        hd = x.shape[-1]
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        c, s = self.rope_cos, self.rope_sin
        if c.shape[0] != x.shape[-2]:
            # variable window at eval (predict on shorter/longer prefix):
            # recompute the local table instead of slicing/mutating the
            # registered buffer (mutation would break CUDA-graph capture)
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.dh, 2,
                                                 device=x.device,
                                                 dtype=torch.float32) / self.dh))
            t = torch.arange(x.shape[-2], dtype=torch.float32,
                             device=x.device)
            f = torch.outer(t, inv)
            c, s = torch.cos(f), torch.sin(f)
        c = c[None, None]
        s = s[None, None]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    def forward(self, x):                       # (B, W, D)
        B, W, D = x.shape
        q = self.q(x).view(B, W, self.heads, self.dh).transpose(1, 2)
        k = self.k(x).view(B, W, self.heads, self.dh).transpose(1, 2)
        v = self.v(x).view(B, W, self.heads, self.dh).transpose(1, 2)
        if self.pos_mode == "rope":
            q, k = self._rope(q), self._rope(k)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if W <= self.mask.shape[0]:
            att = att.masked_fill(~self.mask[:W, :W], float("-inf"))
        else:                       # variable window at eval: local tril
            m = torch.ones(W, W, dtype=torch.bool,
                           device=x.device).tril()
            att = att.masked_fill(~m, float("-inf"))
        ctx = torch.softmax(att, dim=-1) @ v    # convex combos of v: bounded
        ctx = ctx.transpose(1, 2).reshape(B, W, D)
        return self.o(ctx)


class AttLM(nn.Module):
    def __init__(self, input_mode="byte", act="channel", head="phase",
                 window=64, dim=128, depth=6, code_dim=32, n_bytes=8, d_code=32,
                 n_pos=8, skip_alpha=0.7, heads=1, n_attn=1, depth_after=6,
                 tokenizer_path=None, vocab=None, train_allpos=False,
                 pos_mode="harmonic", block_kind="channel", bytes_mat=None):
        super().__init__()
        vocab = vocab or V
        self.vocab = vocab
        self.window, self.input_mode, self.act = window, input_mode, act
        self.train_allpos = train_allpos
        self.iso_lambda = 0.0   # set >0 (e.g. 1e-3) to enable Soft Isometry Prior
        if input_mode == "embedding":
            self.embed = nn.Embedding(vocab, dim)
        elif input_mode == "code":
            # codebook-tied input (weight-tying analog): token -> its phase
            # code row (V x K, shared with the output head) -> lift -> phi.
            # Zero extra vocabulary parameters; the codebook receives
            # gradients from both the input path and the loss.
            self.lift = nn.Linear(code_dim, dim, bias=False)
            nn.init.orthogonal_(self.lift.weight)
        else:   # byte-tuple harmonic token code
            if bytes_mat is not None:            # portable precomputed rows:
                # no tokenizer needed at construction time
                self.register_buffer("bytes_mat", bytes_mat)
            else:
                from tokenizers import Tokenizer
                tk = Tokenizer.from_file(tokenizer_path or TK)
                rows = np.zeros((vocab, n_bytes), dtype=np.int64)
                for i in range(vocab):
                    # id_to_token may be None (added tokens / merge gaps)
                    s = (tk.id_to_token(i) or "").replace("\u0120", " ").replace("\u010a", "\n")
                    b = s.encode("utf-8")[:n_bytes]
                    rows[i, :len(b)] = list(b)
                self.register_buffer("bytes_mat", torch.from_numpy(rows))
            self.omega = nn.Parameter(torch.randn(n_bytes, d_code) * 0.5)
            self.theta = nn.Parameter(torch.zeros(n_bytes, d_code))
            _uniform_phase_init(self.theta)
            self.lift = nn.Linear(d_code, dim, bias=False)
            nn.init.orthogonal_(self.lift.weight)
        # positional encoding: learnable harmonics (FIP-native) or RoPE cache
        self.pos_mode = pos_mode
        if pos_mode == "harmonic":
            self.omega_p = nn.Parameter(torch.randn(n_pos, dim) * 0.2)
            self.theta_p = nn.Parameter(torch.zeros(n_pos, dim))
            _uniform_phase_init(self.theta_p)
        self.attn_layers = nn.ModuleList(
            CausalAttention(dim, window, heads, pos_mode) for _ in range(n_attn))
        self.mix_blocks = nn.ModuleList(BlockV2(dim, block_kind if act == "channel" else act,
                                                skip_alpha) for _ in range(n_attn))
        self.blocks = nn.ModuleList(BlockV2(dim, block_kind if act == "channel" else act,
                                            skip_alpha) for _ in range(depth_after))
        if head == "phase":
            self.decoder = LowRankPhaseBottleneck(dim, vocab, code_dim, "fourier",
                                                  trainable_codes=True)
        else:
            self.head = nn.Linear(dim, vocab)

    def _pe(self, Wv, device):
        pos = torch.arange(Wv, dtype=torch.float32, device=device)
        z = pos.view(-1, 1, 1) * self.omega_p.unsqueeze(0) + self.theta_p.unsqueeze(0)
        return fip_wave(z, GAMMA, BETA).sum(1)      # (W, D)

    def features(self, tokens):
        Wv = tokens.shape[1]
        if self.input_mode == "embedding":
            h = self.embed(tokens)                  # (B, W, D)
        elif self.input_mode == "code":
            c = self.decoder.codes()                # (V, K), shared with output
            code = c[tokens]                        # (B, W, K)
            h = fip_wave(F.linear(code, self.lift.weight), GAMMA, BETA)
        else:
            z = self.bytes_mat[tokens].float().unsqueeze(-1) * self.omega.unsqueeze(0) \
                + self.theta.unsqueeze(0)
            code = fip_wave(z, GAMMA, BETA).sum(2)  # (B, W, dc)
            h = fip_wave(F.linear(code, self.lift.weight), GAMMA, BETA)
        if self.pos_mode == "harmonic":
            h = h + self._pe(Wv, tokens.device)
        # interleaved layer structure: layer i = [attn_i + mix_i, block_i].
        # With n_attn=1 this is identical to the champion order
        # (single attn+mix, then all blocks).
        for i in range(max(len(self.attn_layers), len(self.blocks))):
            if i < len(self.attn_layers):
                h = self.mix_blocks[i](self.attn_layers[i](h))
            if i < len(self.blocks):
                h = self.blocks[i](h)
        if not self.train_allpos:
            h = h[:, -1]                            # last causal position only
        return h                                    # (B, D) or (B, W, D)

    def _iso_reg(self):
        if self.iso_lambda <= 0:
            # zero-tensor on the right device/dtype; blocks may be gated
            # (no .weight since the dead-weight fix), so use any parameter
            p = next(self.blocks.parameters()) if len(self.blocks) else \
                next(self.parameters())
            return p.new_zeros(())
        reg = 0.0
        for blk in list(self.blocks) + list(self.mix_blocks):
            reg = reg + blk.iso_prior()
        return self.iso_lambda * reg

    def loss(self, x, y):
        h = self.features(x)
        if hasattr(self, "decoder"):
            if h.dim() == 3:                        # all-position training: (B, W, D)
                B, Wv, D = h.shape
                loss = self.decoder.loss(h.reshape(B * Wv, D), y.reshape(-1), mode="ce")
            else:
                loss = self.decoder.loss(h, y, mode="ce")
            return loss + self._iso_reg()
        if h.dim() == 3:
            B, Wv, D = h.shape
            logits = self.head(h.reshape(B * Wv, D)).view(B, Wv, -1)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) \
                 + self._iso_reg()
        return F.cross_entropy(self.head(h), y) + self._iso_reg()

    def predict(self, x):
        h = self.features(x)
        if hasattr(self, "decoder"):
            if h.dim() == 3:
                B, Wv, D = h.shape
                return self.decoder.decode(h.reshape(B * Wv, D)).view(B, Wv)
            return self.decoder.decode(h)
        if h.dim() == 3:
            B, Wv, D = h.shape
            return self.head(h.reshape(B * Wv, D)).argmax(-1).view(B, Wv)
        return self.head(h).argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--jobs", type=str, default="ATT1,ATT2,ATT3,ATT4,ATT5")
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

    jobs = [
        ("ATT1 byte+attn+chan-beta+phase", dict(input_mode="byte"), 3e-4),
        ("ATT2 emb+attn+relu+softmax", dict(input_mode="embedding", act="relu", head="softmax"), 3e-3),
        ("ATT3 h4 L1", dict(input_mode="byte", heads=4), 3e-4),
        ("ATT4 h1 L2", dict(input_mode="byte", heads=1, n_attn=2, depth_after=4), 3e-4),
        ("ATT5 h4 L2", dict(input_mode="byte", heads=4, n_attn=2, depth_after=4), 3e-4),
    ]
    want = set(args.jobs.split(","))
    for name, kw, lr in jobs:
        if name.split()[0] not in want: continue
        torch.manual_seed(0)
        model = AttLM(window=W, **kw).to(device)
        if hasattr(model, "decoder"):
            model.decoder.codes()
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.999),
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
        print(f"{name:34s} BEST {best[0]*100:6.2f}% loss {best[1]:.3f} "
              f"params {params/1e6:.2f}M [{' '.join(traj)}] [{time.time()-t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
