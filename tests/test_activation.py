import math

import pytest
import torch

from fipnet.activation import (
    fip_wave,
    fip_tri,
    parabolic_derivative,
    _tri_parts,
    GAMMA,
    BETA,
    PERIOD,
)


def test_output_bounds():
    x = torch.linspace(-200.0, 200.0, 1_000_001)
    phi = fip_wave(x)
    assert phi.min().item() >= -GAMMA - 1e-5
    assert phi.max().item() <= GAMMA + 1e-5


def test_periodicity():
    x = torch.linspace(-10.0, 10.0, 4096)
    assert torch.allclose(fip_wave(x), fip_wave(x + PERIOD / BETA), atol=1e-5)
    assert torch.allclose(fip_wave(x), fip_wave(x + 7 * PERIOD / BETA), atol=1e-5)


def test_values_at_special_points():
    """With theta=0: z = beta*x; u = +1 at z = 3 + 4k, u = -1 at z = 1 + 4k."""
    g, b = GAMMA, BETA
    assert fip_wave(torch.tensor([0.0])).item() == pytest.approx(0.0, abs=1e-6)
    assert fip_wave(torch.tensor([1.0 / b])).item() == pytest.approx(-g, abs=1e-6)
    assert fip_wave(torch.tensor([3.0 / b])).item() == pytest.approx(g, abs=1e-6)
    assert fip_wave(torch.tensor([5.0 / b])).item() == pytest.approx(-g, abs=1e-6)
    assert fip_wave(torch.tensor([7.0 / b])).item() == pytest.approx(g, abs=1e-6)


def test_legacy_single_knob_has_gradient_gain_2_5():
    """beta=1 (original proposal): E[phi'^2] = 5/2, not 1."""
    torch.manual_seed(0)
    theta = (torch.rand(4_000_000) - 0.5) * PERIOD
    x = torch.randn(4_000_000)
    a, u = _tri_parts(x + theta)
    e = (((2.0 * (1.0 - torch.abs(u))) * GAMMA) ** 2).mean().item()
    assert e == pytest.approx(2.5, abs=0.02)


def test_two_param_moments_uniform_phase():
    """With theta ~ U[-2,2] per neuron, z is uniform on the torus for ANY
    input distribution: E[phi^2] = 1 and E[phi'^2] = 1 simultaneously."""
    torch.manual_seed(0)
    n = 4_000_000
    x = torch.randn(n)
    theta = (torch.rand(n) - 0.5) * PERIOD
    z = BETA * x + theta
    a, u = _tri_parts(z)
    e_phi2 = ((GAMMA * u * (2 - torch.abs(u))) ** 2).mean().item()
    e_grad2 = ((GAMMA * BETA * 2.0 * (1.0 - torch.abs(u))) ** 2).mean().item()
    assert e_phi2 == pytest.approx(1.0, abs=0.01)
    assert e_grad2 == pytest.approx(1.0, abs=0.01)


def test_analytic_grad_matches_finite_difference():
    # avoid the |u| cusp where the derivative jumps
    x = torch.arange(0.05, 2.9, 0.137, dtype=torch.float64, requires_grad=True)
    out = fip_wave(x).sum()
    out.backward()
    analytic = x.grad

    h = 1e-6
    x0 = torch.arange(0.05, 2.9, 0.137, dtype=torch.float64)
    numeric = (fip_wave(x0 + h) - fip_wave(x0 - h)) / (2 * h)

    assert torch.allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_sign_flip_of_gradient():
    """phi' changes sign across z = 1 + 2k (self-regulation property)."""
    left = parabolic_derivative(torch.tensor([0.2 / BETA])).item()
    right = parabolic_derivative(torch.tensor([1.8 / BETA])).item()
    assert left * right < 0  # opposite signs on either side of z = 1


def test_triangular_derivative_is_pm1():
    torch.manual_seed(0)
    x = torch.randn(1_000_000) * 7
    y = fip_tri(x)
    y.sum().backward() if y.requires_grad else None
    # analytic check away from cusps: derivative magnitude exactly 1
    x2 = torch.linspace(0.05, 10.0, 5000, dtype=torch.float64, requires_grad=True)
    fip_tri(x2).sum().backward()
    assert torch.allclose(x2.grad.abs(), torch.ones_like(x2.grad), atol=1e-6)


def test_triangular_moment():
    """E[phi_tri^2] = 1 under theta ~ U[-2,2], any input distribution."""
    torch.manual_seed(1)
    n = 4_000_000
    x = torch.randn(n)
    theta = (torch.rand(n) - 0.5) * PERIOD
    a, u = _tri_parts(x / math.sqrt(3.0) + theta)
    e = ((math.sqrt(3.0) * u) ** 2).mean().item()
    assert e == pytest.approx(1.0, abs=0.01)
