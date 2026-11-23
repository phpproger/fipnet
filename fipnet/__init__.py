"""FIP-Net: Fast Isometric Periodic Network.

Reference implementation of the normalization-free, embedding-free
architecture with two-parametric parabolic and strictly isometric
triangular periodic activations.
"""

from .activation import (
    fip_wave,
    fip_tri,
    parabolic_derivative,
    _tri_parts,
    GAMMA,
    BETA,
    PERIOD,
)
from .model import (
    HarmonicEncoder,
    IsometricBlock,
    LowRankPhaseBottleneck,
    InterferometricDecoder,
    FIPNet,
    FIPMnist,
    MlpBaseline,
)

__all__ = [
    "fip_wave",
    "fip_tri",
    "parabolic_derivative",
    "GAMMA",
    "BETA",
    "PERIOD",
    "HarmonicEncoder",
    "IsometricBlock",
    "LowRankPhaseBottleneck",
    "InterferometricDecoder",
    "FIPNet",
    "FIPMnist",
    "MlpBaseline",
]
