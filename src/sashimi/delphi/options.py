"""DelPhi-specific knobs, and the mapping from solver-neutral concepts onto them.

Everything here is DelPhi vocabulary, which is why it lives under
`sashimi.delphi` and not in the protocol — the same rule `sashimi.apbs.options`
follows.

**The surface-model mapping table.** ROADMAP.md section 14 left "which
solver-neutral enum members exist, and how each backend maps them" open, noting
that DelPhi's Gaussian dielectric has no APBS equivalent so the enum could not
just be APBS's set renamed. Measured against the two real backends:

| `SurfaceModel`       | APBS            | DelPhi C++   | pyDelPhi              |
|----------------------|-----------------|--------------|-----------------------|
| `MOLECULAR`          | `srfm mol`      | `prbrad > 0` | `surfmethod vdw`, `prbrad > 0` |
| `SMOOTHED_MOLECULAR` | `srfm smol`     | **no**       | **no**                |
| `VAN_DER_WAALS`      | `mol`, `srad 0` | `prbrad 0`   | **no** (see below)    |
| `GAUSSIAN`           | **no**          | `gaussian 1` | `surfmethod gaussian` |

The pyDelPhi row is the counter-intuitive one, and it was got wrong here once.
`surfmethod=vdw` names the *construction* — roll a probe over van der Waals
spheres, the algorithm its `vdwms` module is named for — not the resulting
surface. With a probe it produces the molecular surface, and it produces the
C++ build's `molecular` answer exactly: -84.33 kT on ALA-GLY and -92.22 kT on
the Born ion, identical to the last printed digit on matched grids. Its
probe-dependence (-84.33 kT at 1.4 A against -86.88 at 0.5 A) is the molecular
surface behaving correctly, not evidence that it is something else.

Two consequences:

- `SMOOTHED_MOLECULAR` — harmonic dielectric averaging, and sashimi's *default*
  — is APBS-only. A DelPhi solve at defaults therefore raises
  `UnsupportedRequest` rather than quietly substituting `MOLECULAR`, which
  would move a dipeptide's energy by more than 2,000x the corpus tolerance.
- `VAN_DER_WAALS` is the probe-free union of atomic spheres, which pyDelPhi
  cannot currently deliver: `prbrad=0` is the natural limit of its own `vdw`
  method and aborts it with a numba `TypingError`, in 0.2.0 and 0.3.0 alike.
  That is a bug rather than a modelling difference, so the model is declined
  until it is fixed rather than mapped onto something adjacent.

`GAUSSIAN` is supported by both DelPhi flavours and by neither APBS nor any
closed form. **No equivalent request has been established across the two
flavours**: matching `sigma` and `srfcut` still leaves them at -152.43 kT and
-38.90 kT on the same ALA-GLY grid, and pyDelPhi's answer does not move with
either parameter, so those are evidently not the corresponding knobs. That is a
statement about an unfinished comparison, not evidence the models differ.
`capabilities` marks it unvalidated so no number from it is read as cross-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

from sashimi.delphi.discover import DelphiFlavour
from sashimi.errors import UnsupportedRequest
from sashimi.protocol import Equation, SurfaceModel

__all__ = ["SUPPORTED_SURFACES", "DelphiOptions", "ResolvedSurface", "resolve_surface"]

# Which surface models each flavour can actually produce. The C++ program has no
# surface-method keyword at all — it selects between molecular and van der Waals
# by whether the probe radius is zero, the same trick APBS uses for `srad 0`.
SUPPORTED_SURFACES: dict[DelphiFlavour, frozenset[SurfaceModel]] = {
    DelphiFlavour.CPP: frozenset(
        {SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS, SurfaceModel.GAUSSIAN}
    ),
    DelphiFlavour.PYDELPHI: frozenset({SurfaceModel.MOLECULAR, SurfaceModel.GAUSSIAN}),
}

# Surface models no cross-solver comparison should be built on. A Gaussian
# dielectric has no APBS counterpart and no closed form, and no equivalent
# request has been established across the two DelPhi flavours; `capabilities`
# reports that rather than leaving it to be discovered.
UNVALIDATED_SURFACES: frozenset[SurfaceModel] = frozenset({SurfaceModel.GAUSSIAN})

# pyDelPhi names its surface methods; the C++ program infers them. `vdw` is the
# probe-rolling construction, so with a probe it *is* the molecular surface —
# see the module docstring, and note that `VAN_DER_WAALS` is deliberately absent
# because the `prbrad=0` that would produce it crashes.
_PY_SURFACE_KEYWORD: dict[SurfaceModel, str] = {
    SurfaceModel.MOLECULAR: "vdw",
    SurfaceModel.GAUSSIAN: "gaussian",
}


@dataclass(frozen=True)
class ResolvedSurface:
    """What a surface model became, in DelPhi's own terms.

    Carried into provenance whole, because "which surface did this actually
    run" is the first question any cross-solver comparison has to answer.
    """

    probe_radius: float
    gaussian: bool
    keyword: str | None = None  # pyDelPhi's `surfmethod`; None for the C++ program


@dataclass(frozen=True)
class DelphiOptions:
    """Backend-specific overrides. Defaults reproduce sashimi's documented behaviour.

    `linear_iterations` and `max_delta_phi` are not tuning knobs, they are a
    correctness requirement: DelPhi C++ 8.6 defaults to `linit=0` and
    `maxc=0.0`, and a solve that inherits both never terminates. Measured on a
    Born ion, it passed 1.39 million iterations with residuals already at
    3e-16 — machine epsilon — because the convergence test compares against a
    threshold of zero that no finite residual can meet. sashimi always writes
    both values.
    """

    linear_iterations: int = 1000  # DelPhi `linit`
    max_delta_phi: float = 1e-4  # DelPhi `maxc`, kT/e
    relaxation: float | None = None  # DelPhi `relpar`; None leaves it automatic
    gaussian_sigma: float = 0.93  # DelPhi `sigma`, only read for a Gaussian surface
    gaussian_cutoff: float = 20.0  # DelPhi `srfcut`, likewise

    def __post_init__(self) -> None:
        if self.linear_iterations <= 0:
            raise ValueError(f"linear_iterations must be positive, got {self.linear_iterations}")
        if self.max_delta_phi <= 0:
            raise ValueError(
                f"max_delta_phi must be positive, got {self.max_delta_phi}. DelPhi treats "
                "zero as a threshold no residual can reach, and iterates forever."
            )


def resolve_surface(
    model: SurfaceModel,
    probe_radius: float,
    flavour: DelphiFlavour,
) -> ResolvedSurface:
    """Map a portable surface model onto DelPhi's parameters for this flavour.

    Raises `UnsupportedRequest` naming both what this flavour supports and,
    where one exists, the backend that does support what was asked — an agent
    that learns `SMOOTHED_MOLECULAR` is unavailable here should not have to
    discover separately that APBS has it.
    """
    supported = SUPPORTED_SURFACES[flavour]
    if model not in supported:
        names = ", ".join(sorted(m.value for m in supported))
        hint = ""
        if model is SurfaceModel.SMOOTHED_MOLECULAR:
            hint = (
                " Harmonic dielectric smoothing is an APBS feature with no DelPhi "
                "equivalent; use SurfaceModel.MOLECULAR for a comparable calculation, "
                "and expect a different number rather than the same one."
            )
        elif flavour is DelphiFlavour.PYDELPHI and model is SurfaceModel.VAN_DER_WAALS:
            hint = (
                " pyDelPhi's 'vdw' method is the probe-rolling construction, so it gives the "
                "molecular surface rather than the probe-free one this model names. The "
                "setting that would produce it, prbrad=0, aborts pyDelPhi inside numba "
                "(0.2.0 and 0.3.0 alike). Use the C++ DelPhi build for a van der Waals "
                "boundary, or SurfaceModel.MOLECULAR here."
            )
        raise UnsupportedRequest(
            f"{flavour.value} has no equivalent of the {model.value!r} surface model. "
            f"It supports: {names}.{hint}"
        )

    if model is SurfaceModel.GAUSSIAN:
        # A Gaussian dielectric has no sharp boundary, so a probe radius is
        # meaningless rather than merely unused.
        return ResolvedSurface(probe_radius=0.0, gaussian=True, keyword=_keyword(model, flavour))

    # Van der Waals is the molecular surface with the probe collapsed — the same
    # identity APBS expresses as `srad 0`.
    radius = 0.0 if model is SurfaceModel.VAN_DER_WAALS else probe_radius
    return ResolvedSurface(probe_radius=radius, gaussian=False, keyword=_keyword(model, flavour))


def _keyword(model: SurfaceModel, flavour: DelphiFlavour) -> str | None:
    if flavour is DelphiFlavour.PYDELPHI:
        return _PY_SURFACE_KEYWORD[model]
    return None


def check_equation(equation: Equation, flavour: DelphiFlavour) -> None:
    """DelPhi solves both equations; sashimi ships only the linear path.

    Kept here rather than inlined in the backend so the reason travels with the
    other capability statements: this is a sashimi limit, not a DelPhi one, and
    lifting it needs the NPBE-versus-LPBE comparability rule of ROADMAP.md
    section 14 rather than a keyword.
    """
    if equation is not Equation.LINEAR:
        raise UnsupportedRequest(
            f"sashimi can express the {equation.value} equation but does not yet solve it. "
            f"{flavour.value} implements it natively, so enabling it is a question of "
            "settling how nonlinear and linear energies may be compared "
            "(ROADMAP.md section 14), not of backend support. Use Equation.LINEAR."
        )
