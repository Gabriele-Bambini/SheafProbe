"""Default hyperparameters for SheafProbe (small, CPU-only, finishes in minutes).

Two presets are exposed:
- ``DEFAULT`` : the full small config used for the headline runs.
- ``QUICK``   : a tiny smoke config for CI / `--quick` (fewer samples, fewer epochs).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict


@dataclass(frozen=True)
class Config:
    """Frozen hyperparameter bundle.

    Attributes mirror the CONTRACT knobs; all kept small so everything runs on CPU
    in minutes (length<=68, <=600 samples, k<=8, <=200 epochs).
    """

    LENGTH: int = 68          # nucleotides per synthetic molecule
    N_SAMPLES: int = 600      # dataset size
    K: int = 8                # sheaf stalk dimension
    N_LAYERS: int = 4         # sheaf-diffusion steps
    EPOCHS: int = 150         # training epochs
    LR: float = 1e-2          # Adam learning rate
    LAM: float = 1.0          # weight of the Dirichlet (E*) term in the loss
    EPS: float = 0.3          # diffusion step size
    SEED: int = 0             # global RNG seed
    FRAC_MULTISTATE: float = 0.5   # fraction of multi-state molecules
    NOISE: float = 0.15            # Gaussian reactivity noise std

    def as_dict(self) -> Dict[str, object]:
        """Return a plain JSON-able dict of the config."""
        return asdict(self)


# Full small config (the default for headline runs).
DEFAULT = Config()

# Tiny fast variant for smoke tests / `--quick`.
QUICK = replace(DEFAULT, N_SAMPLES=120, EPOCHS=40)


def get_config(quick: bool = False) -> Config:
    """Return the QUICK config when ``quick`` is True, else the DEFAULT config."""
    return QUICK if quick else DEFAULT
