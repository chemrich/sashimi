"""TABI-PB's knobs, and the surface mapping onto them.

The mapping is short because a boundary-element solver has one surface: the
triangulated dielectric interface it integrates over. There is no dielectric
smoothing to choose, no charge discretization, and no grid — which is the point
of adding this backend at all.

| `SurfaceModel`       | TABI-PB      |
|----------------------|--------------|
| `MOLECULAR`          | `mesh ses`   |
| `SMOOTHED_MOLECULAR` | no           |
| `VAN_DER_WAALS`      | no, in practice |
| `GAUSSIAN`           | no           |

`SMOOTHED_MOLECULAR` is APBS's harmonic averaging, meaningless without a grid.
`GAUSSIAN` is a volumetric dielectric, meaningless without one too. Neither is a
gap to be filled; they are grid concepts, and a BEM solver having no equivalent
is the protocol behaving correctly rather than a backend falling short.

`VAN_DER_WAALS` would be `srad 0`, and NanoShaper does not return from it in any
reasonable time on a dipeptide, so it is declined rather than offered as a trap.

TABI-PB's other mesh type, Edelsbrunner's `skin` surface, has no solver-neutral
name and no counterpart in either FD backend. It sits behind `TabipbOptions`,
where asking for it is explicit — the same treatment `spl2` gets in
`sashimi.apbs.options`.
"""

from __future__ import annotations

from dataclasses import dataclass

from sashimi.errors import UnsupportedRequest
from sashimi.protocol import SurfaceModel

__all__ = ["SUPPORTED_SURFACES", "TabipbOptions", "resolve_mesh"]

SUPPORTED_SURFACES: frozenset[SurfaceModel] = frozenset({SurfaceModel.MOLECULAR})

_MESH_KEYWORD: dict[SurfaceModel, str] = {SurfaceModel.MOLECULAR: "ses"}


@dataclass(frozen=True)
class TabipbOptions:
    """Backend-specific overrides; defaults reproduce sashimi's documented behaviour.

    The tree parameters control the treecode that makes TABI-PB O(N) rather
    than O(N^2). They are accuracy-versus-speed knobs, not physics: `tree_theta`
    of 0 disables the far-field approximation entirely, which is the exact and
    slowest setting, and is the default here because a wrong-but-fast default
    would show up as a solver disagreement in `sashimi validate` and be blamed
    on the wrong thing.
    """

    mesh_override: str | None = None  # "skin", the surface with no portable name
    tree_degree: int = 1
    tree_theta: float = 0.0
    tree_max_per_leaf: int = 50
    precondition: bool = True

    def __post_init__(self) -> None:
        if self.mesh_override is not None and self.mesh_override not in ("ses", "skin"):
            raise ValueError(f"mesh_override must be 'ses' or 'skin', got {self.mesh_override!r}")
        if not 0.0 <= self.tree_theta < 1.0:
            raise ValueError(f"tree_theta must be in [0, 1), got {self.tree_theta}")


def resolve_mesh(model: SurfaceModel, options: TabipbOptions) -> str:
    """Map a portable surface model onto TABI-PB's `mesh` keyword."""
    if options.mesh_override is not None:
        return options.mesh_override

    keyword = _MESH_KEYWORD.get(model)
    if keyword is None:
        hint = ""
        if model in (SurfaceModel.SMOOTHED_MOLECULAR, SurfaceModel.GAUSSIAN):
            hint = (
                " That model describes how a dielectric varies across a *grid*, which a "
                "boundary-element solver does not have; it is not a missing feature."
            )
        elif model is SurfaceModel.VAN_DER_WAALS:
            hint = (
                " It would be a zero probe radius, which the NanoShaper triangulation does "
                "not complete in reasonable time."
            )
        raise UnsupportedRequest(
            f"TABI-PB has no equivalent of the {model.value!r} surface model. "
            f"It supports: {', '.join(sorted(m.value for m in SUPPORTED_SURFACES))}.{hint}"
        )
    return keyword
