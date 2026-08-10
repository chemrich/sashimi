"""APBS-specific knobs, and the mapping from solver-neutral concepts onto them.

Everything here is APBS vocabulary, which is why it lives under `sashimi.apbs`
and not in the protocol. `ApbsOptions` is the escape hatch ROADMAP.md §14 Q2
decided on: the portable `SurfaceModel` covers what every backend can mean, and
anything beyond it has to be asked for explicitly, here, by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from sashimi.errors import UnsupportedRequest
from sashimi.protocol import Equation, SurfaceModel

__all__ = ["EQUATION_KEYWORD", "SURFACE_KEYWORD", "ApbsOptions", "resolve_surface"]

# Solver-neutral surface model -> APBS `srfm` keyword.
SURFACE_KEYWORD: dict[SurfaceModel, str] = {
    SurfaceModel.MOLECULAR: "mol",
    SurfaceModel.SMOOTHED_MOLECULAR: "smol",
    # APBS expresses a van der Waals boundary as the molecular surface with a
    # zero-radius probe rather than as its own srfm value; the backend sets
    # `srad 0` alongside this. See `resolve_surface`.
    SurfaceModel.VAN_DER_WAALS: "mol",
}

EQUATION_KEYWORD: dict[Equation, str] = {
    Equation.LINEAR: "lpbe",
    Equation.NONLINEAR: "npbe",
}


@dataclass(frozen=True)
class ApbsOptions:
    """Backend-specific overrides. Defaults reproduce sashimi's documented behaviour.

    `srfm_override` reaches the spline surfaces (`spl2`, `spl4`) that the
    portable `SurfaceModel` deliberately omits. They exist to give smooth
    derivatives for *force* calculations; using one for solvation energy is a
    misuse that moved a dipeptide's energy by 25% in testing. Reaching for this
    field is how you say you meant it.
    """

    srfm_override: str | None = None
    spline_window: float = 0.3  # APBS `swin`, only meaningful for spline surfaces
    charge_discretization: str = "spl4"  # APBS `chgm`
    boundary_condition: str = "sdh"  # APBS `bcfl`
    surface_density: float = 10.0  # APBS `sdens`


def resolve_surface(
    model: SurfaceModel, probe_radius: float, options: ApbsOptions
) -> tuple[str, float]:
    """Map a portable surface model onto (`srfm`, `srad`).

    Returns the probe radius too, because van der Waals is not a distinct APBS
    surface method — it is the molecular surface with the probe collapsed.
    """
    if options.srfm_override is not None:
        return options.srfm_override, probe_radius

    if model is SurfaceModel.VAN_DER_WAALS:
        return SURFACE_KEYWORD[model], 0.0

    keyword = SURFACE_KEYWORD.get(model)
    if keyword is None:
        supported = ", ".join(sorted(m.value for m in SURFACE_KEYWORD))
        raise UnsupportedRequest(
            f"APBS has no equivalent of the {model.value!r} surface model. "
            f"It supports: {supported}. "
            "A Gaussian dielectric is DelPhi's model and has no APBS mapping."
        )
    return keyword, probe_radius
