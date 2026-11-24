"""Fast Isometric Periodic activations.

Two-parametric parabolic wave (period T=4):
    z      = beta * x + theta
    u(z)   = 1 - 2*|2*frac((z-1)/4) - 1|          triangular wave in [-1, 1]
    f(u)   = 2u - u|u|                             parabolic smoothing
    phi(x) = gamma * f(u(z)),                      |phi| <= gamma

Analytic derivative (a.s.):
    dphi/dx = gamma * beta * 2 * (1 - |u|) * (-sign(2f - 1)),   |du/dz| = 1

Moment matching under uniform phase u ~ U[-1, 1] (guaranteed at init by
theta ~ U[-2, 2] per neuron, by shift-invariance on the torus R/4Z):
    E[phi^2]   = gamma^2 * 8/15        -> gamma = sqrt(15/8) ~ 1.3693
    E[phi'^2]  = gamma^2 beta^2 * 4/3  -> beta  = sqrt(2/5)  ~ 0.6325
Both conditions hold simultaneously for the pair (gamma, beta).
Legacy single-knob config (beta=1) gives E[phi'^2] = 5/2.

Strictly isometric triangular variant:
    phi_tri(x) = sqrt(3) * u(x/sqrt(3) + theta)
    E[phi_tri^2] = 1,  phi_tri' = +-1 (a.s.)  ->  D_l = diag(+-1) orthogonal,
    so prod(D_l W_l) preserves the full singular-value spectrum exactly.
"""

import math

import torch

GAMMA = math.sqrt(15.0 / 8.0)     # unit output variance: E[phi^2] = 1
BETA = math.sqrt(2.0 / 5.0)       # unit gradient gain:   E[phi'^2] = 1
# Legacy single-parameter value (beta=1): E[phi'^2] = gamma^2 * 4/3 = 5/2.
LEGACY_GRAD_GAIN = 2.5

PERIOD = 4.0


def _tri_parts(z):
    """Return (a, u) where a = 2*frac((z-1)/4) - 1 in [-1,1), u = 1 - 2|a|."""
    s = (z - 1.0) / PERIOD
    f = s - torch.floor(s)              # fractional part in [0, 1)
    a = 2.0 * f - 1.0                   # sawtooth in [-1, 1)
    u = 1.0 - 2.0 * torch.abs(a)        # triangular wave in [-1, 1]
    return a, u


class _FIPParabolic(torch.autograd.Function):
    """phi(x) = gamma * f(u(beta*x + theta)) with analytic backward."""

    @staticmethod
    def forward(ctx, x, gamma, beta, theta):
        a, u = _tri_parts(beta * x + theta)
        phi = gamma * u * (2.0 - torch.abs(u))
        ctx.save_for_backward(a)            # u is recomputed from a in backward
        ctx.gamma_beta = gamma * beta
        return phi

    @staticmethod
    def backward(ctx, grad_out):
        (a,) = ctx.saved_tensors
        u = 1.0 - 2.0 * torch.abs(a)        # recompute, halves saved memory
        dphi_du = 2.0 * (1.0 - torch.abs(u))
        du_dz = -torch.sign(a)              # magnitude 1 a.s.; 0 at cusps
        return grad_out * ctx.gamma_beta * dphi_du * du_dz, None, None, None


class _FIPTriangular(torch.autograd.Function):
    """phi_tri(x) = sqrt(3) * u(x/sqrt(3) + theta); derivative = +-1 a.s."""

    @staticmethod
    def forward(ctx, x, theta):
        a, u = _tri_parts(x / math.sqrt(3.0) + theta)
        ctx.save_for_backward(a)
        return math.sqrt(3.0) * u

    @staticmethod
    def backward(ctx, grad_out):
        (a,) = ctx.saved_tensors
        # dphi/dx = sqrt(3) * du/dz * (1/sqrt(3)) = -sign(a)
        return grad_out * (-torch.sign(a)), None


def _fip_wave_eager(x, gamma=GAMMA, beta=BETA, theta=0.0):
    """Plain-op form for torch.compile: same math as _FIPParabolic, but as
    ~5 fused-able elementwise ops instead of a custom autograd Function
    (inductor merges them into one kernel; the custom Function blocks
    fusion). Cusps (|u|=1) have zero derivative - same as the analytic one.
    """
    z = beta * x + theta
    s = (z - 1.0) / PERIOD
    a = 2.0 * (s - torch.floor(s)) - 1.0
    u = 1.0 - 2.0 * torch.abs(a)
    return gamma * u * (2.0 - torch.abs(u))


def fip_wave(x, gamma=GAMMA, beta=BETA, theta=0.0):
    """Two-parametric Fast Isometric Parabolic Wave."""
    return _FIPParabolic.apply(x, gamma, beta, theta)


def fip_wave_c(x, gamma=GAMMA, beta=BETA, theta=0.0):
    """Compile-friendly entry: eager ops (inductor fuses these)."""
    return _fip_wave_eager(x, gamma, beta, theta)


def fip_tri(x, theta=0.0):
    """Strictly isometric triangular wave: phi' = +-1 a.s."""
    return _FIPTriangular.apply(x, theta)


def parabolic_derivative(x, gamma=GAMMA, beta=BETA, theta=0.0):
    """Analytic derivative for diagnostics."""
    with torch.no_grad():
        a, u = _tri_parts(beta * x + theta)
        return gamma * beta * 2.0 * (1.0 - torch.abs(u)) * (-torch.sign(a))
