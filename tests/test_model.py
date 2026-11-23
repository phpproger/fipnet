import math

import pytest
import torch
import torch.nn.functional as F

from fipnet.model import (
    HarmonicEncoder,
    IsometricBlock,
    LowRankPhaseBottleneck,
    FIPNet,
    FIPMnist,
    MlpBaseline,
)
from fipnet.activation import fip_wave, fip_tri, GAMMA, BETA


def test_no_embedding_table():
    """No parameter of shape V x D anywhere in the model."""
    model = FIPNet(vocab=1000, dim=64, depth=4, code_dim=10)
    for name, p in model.named_parameters():
        assert not (p.dim() == 2 and p.shape[0] == 1000), f"embedding-like param: {name}"


def test_parameter_count_independent_of_vocab():
    params_small = sum(p.numel() for p in FIPNet(vocab=32, dim=64).parameters())
    params_large = sum(p.numel() for p in FIPNet(vocab=5000, dim=64).parameters())
    assert params_small == params_large


def test_orthogonal_initialization():
    torch.manual_seed(0)
    block = IsometricBlock(128)
    eye = torch.eye(128)
    assert torch.allclose(block.weight.t() @ block.weight, eye, atol=1e-4)


def test_bias_uniform_phase_init():
    torch.manual_seed(0)
    block = IsometricBlock(4096)
    assert block.bias.min() >= -2.0 and block.bias.max() <= 2.0
    mean = block.bias.mean().item()
    assert abs(mean) < 0.1  # centered on the period


def test_jacobian_spectrum_parabolic_vs_triangular():
    """Full singular-value spectrum of the L-layer Jacobian.

    Parabolic (gamma, beta): E[sigma_i^2] = 1 on average, but the spectrum
    spreads anisotropically (Lyapunov effect): sigma_max >> 1, sigma_min << 1.
    Triangular: phi' = +-1 -> D_l orthogonal -> every sigma_i exactly 1.
    """
    torch.manual_seed(0)
    L, D = 10, 256

    def jacobian_spectra(activation):
        blocks = [IsometricBlock(D, activation=activation) for _ in range(L)]
        h = torch.randn(D)
        J = None
        for blk in blocks:
            z = F.linear(h, blk.weight, blk.bias)
            # analytic derivative diagonal
            a = torch.remainder((z - 1.0) / 4.0, 1.0) * 2.0 - 1.0
            u = 1.0 - 2.0 * torch.abs(a)
            if activation == "parabolic":
                d = blk.gamma * blk.beta * 2.0 * (1.0 - torch.abs(u)) * (-torch.sign(a))
            else:
                d = -torch.sign(a)
            DW = d.unsqueeze(-1) * blk.weight
            J = DW if J is None else DW @ J
            h = (fip_wave(z, blk.gamma, blk.beta) if activation == "parabolic"
                 else fip_tri(z))
        return torch.linalg.svdvals(J)

    s_par = jacobian_spectra("parabolic")
    s_tri = jacobian_spectra("triangular")

    assert torch.allclose(s_tri, torch.ones_like(s_tri), atol=1e-4)
    assert s_par.max().item() > 1.5            # some directions explode
    assert s_par.min().item() < 0.2            # most directions decay
    ms = (s_par ** 2).mean().item()            # mean square ~ 1
    assert 0.5 < ms < 2.0


def test_gradient_flow_deep_stack():
    """Backward amplification toward the input over a deep stack.

    Triangular blocks: exact isometry -> growth exactly 1.0.
    Parabolic (gamma, beta): mean-square 1, log-normal spread.
    Legacy beta=1 config (E[phi'^2]=5/2): ~1.58x per layer -> explosion.
    """
    torch.manual_seed(0)

    def run(make_block, depth=32, dim=64):
        blocks = torch.nn.ModuleList(make_block() for _ in range(depth))
        h0 = torch.randn(512, dim)
        hs = []
        h = h0
        for b in blocks:
            h = b(h)
            h.retain_grad()
            hs.append(h)
        loss = (h @ torch.randn(dim)).mean()
        loss.backward()
        norms = torch.tensor([hi.grad.norm().item() for hi in hs])
        blocks.zero_grad(set_to_none=True)
        return norms  # hs[0] deepest backward (nearest input)

    n_tri = run(lambda: IsometricBlock(64, activation="triangular"))
    n_par = run(lambda: IsometricBlock(64, gamma=GAMMA, beta=BETA))
    n_leg = run(lambda: IsometricBlock(64, gamma=GAMMA, beta=1.0))

    growth_tri = (n_tri[0] / n_tri[-1]).item()
    growth_par = (n_par[0] / n_par[-1]).item()
    growth_leg = (n_leg[0] / n_leg[-1]).item()

    assert abs(growth_tri - 1.0) < 1e-3        # strict isometry
    assert 0.2 < growth_par < 5.0              # near-isometric, spread only
    assert growth_leg > growth_par * 100       # legacy config explodes


def test_codes_are_distinct():
    # binary vernier codes differ by few bits (high cosine sim is inherent);
    # quad codes with enough pairs stay well-separated and unit-norm
    for mode, thr, kd in (("binary", 0.999, 6), ("log", 0.96, 6), ("quad", 0.99, 8)):
        dec = LowRankPhaseBottleneck(dim=32, vocab=64, code_dim=kd, mode=mode,
                                     omega_min=0.4477, omega_max=9.13)
        codes = dec.codes()
        assert torch.allclose(codes.norm(dim=-1), torch.ones(64), atol=1e-5)
        c = codes
        sim = c @ c.t()
        sim.fill_diagonal_(-2)
        assert sim.max().item() < thr


def test_fourier_codebook_separation():
    """Optimized zero-mean fourier codebook for large vocab: the raw
    sinusoidal construction has near-parallel pairs at V >> K^2
    (max cosine 0.996 at V=2048, K=16); refinement must fix that."""
    dec = LowRankPhaseBottleneck(dim=32, vocab=512, code_dim=16, mode="fourier")
    c = dec.codes()
    assert torch.allclose(c.norm(dim=-1), torch.ones(512), atol=1e-4)
    G = c @ c.t()
    G.fill_diagonal_(torch.nan)
    G = G[~torch.isnan(G)]
    assert G.max().item() < 0.65


def test_quad_codes_geometry():
    """Quadrature codebook: unit norms and bounded class similarity."""
    dec = LowRankPhaseBottleneck(dim=32, vocab=10, code_dim=8, mode="quad")
    c = dec.codes()
    assert torch.allclose(c.norm(dim=-1), torch.ones(10), atol=1e-5)
    sim = c @ c.t()
    sim.fill_diagonal_(-2)
    assert sim.max().item() < 0.6          # vernier binary was 0.94


def test_loss_modes_stable():
    """ce and mse losses must not blow up even from a zero-initialized head."""
    torch.manual_seed(0)
    for mode in ("ce", "mse", "cos"):
        dec = LowRankPhaseBottleneck(dim=32, vocab=10, code_dim=8, mode="quad")
        dec.head.weight.data.zero_()          # |z_hat| = 0: cos loss singular
        h = torch.randn(64, 32)
        y = torch.randint(0, 10, (64,))
        loss = dec.loss(h, y, mode=mode)
        loss.backward()
        assert torch.isfinite(loss), mode
        gn = dec.head.weight.grad.norm().item()
        if mode == "cos":
            assert gn > 1e4                    # documented 1/|z_hat| singularity
        else:
            assert gn < 1e3, (mode, gn)        # ce/mse stay bounded


def test_resonance_recovers_exact_code():
    dec = LowRankPhaseBottleneck(dim=8, vocab=32, code_dim=6, mode="binary")
    codes = dec.codes()
    assert (torch.argmax(codes @ codes.t(), dim=-1) == torch.arange(32)).all()


def test_unwrap_decoding_identity():
    """O(K) phase unwrapping must decode every exact code correctly."""
    for vocab in (16, 32, 64):
        dec = LowRankPhaseBottleneck(dim=8, vocab=vocab, code_dim=6, mode="binary")
        codes = dec.codes()
        assert (dec.unwrap_from_z(codes) == torch.arange(vocab)).all()


def test_unwrap_matches_resonance_on_noisy_codes():
    torch.manual_seed(0)
    dec = LowRankPhaseBottleneck(dim=8, vocab=16, code_dim=6, mode="binary")
    base = dec.codes()
    codes = base + 0.02 * torch.randn_like(base)
    assert (dec.unwrap_from_z(codes) == torch.argmax(codes @ base.t(), -1)).all()


def test_fipnet_loss_backward_and_predict():
    torch.manual_seed(0)
    model = FIPNet(vocab=32, dim=64, depth=4, code_dim=6)
    tokens = torch.randint(0, 32, (128,))
    loss = model.loss(tokens)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    preds = model.predict(tokens)
    assert preds.shape == (128,) and preds.dtype == torch.long


def test_encoder_learnable_phase_receives_grad():
    model = FIPNet(vocab=32, dim=64, depth=2, code_dim=6)
    model.loss(torch.randint(0, 32, (64,))).backward()
    assert model.encoder.theta.grad is not None
    assert model.encoder.theta.grad.abs().sum() > 0


def test_fip_mnist_shapes_and_backward():
    torch.manual_seed(0)
    for act in ("parabolic", "triangular"):
        model = FIPMnist(n_classes=10, dim=32, depth=2, code_dim=4, activation=act)
        images = torch.rand(16, 1, 28, 28)
        targets = torch.randint(0, 10, (16,))
        loss = model.loss(images, targets)
        loss.backward()
        preds = model.predict(images)
        assert preds.shape == (16,)
        assert torch.isfinite(loss)


def test_fip_mnist_linear_mode():
    model = FIPMnist(n_classes=10, dim=32, depth=2, code_dim=4,
                     input_mode="linear")
    images = torch.rand(8, 1, 28, 28)
    loss = model.loss(images, torch.randint(0, 10, (8,)))
    assert torch.isfinite(loss)


def test_baseline_runs():
    model = MlpBaseline(n_classes=10, dim=32, depth=2)
    images = torch.rand(8, 1, 28, 28)
    loss = model.loss(images, torch.randint(0, 10, (8,)))
    loss.backward()
    assert model.predict(images).shape == (8,)
