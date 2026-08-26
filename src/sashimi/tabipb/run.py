"""Running TABI-PB.

Two things here are not shared with the finite-difference backends.

**The mesher has to be on PATH.** `tabipb` invokes NanoShaper by bare name
through a shell, so an installed-but-not-on-PATH mesher fails with
`sh: NanoShaper: command not found` followed by an uncaught C++ exception. The
subprocess environment therefore gets the mesher's directory prepended, which
also means sashimi never depends on the caller's PATH being arranged correctly.

**A solute can be too small to triangulate.** NanoShaper refuses fewer than four
atoms — there is no surface to mesh — so the Born ion, the one case in this
project with a closed-form answer, cannot be run through a BEM solver at all
without adding dummy atoms. That is checked up front and raised as an
`InputError`, because the alternative is `stoul: no conversion` from deep inside
a C++ binary.

Success is verified structurally, as with every other backend: the VTK must
exist and parse, and the energy must have been printed. TABI-PB exits 0 after
the mesher fails.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from sashimi.errors import InputError, SolverCrash
from sashimi.protocol import FloatArray, SurfacePotential
from sashimi.tabipb.discover import TabipbBinary
from sashimi.tabipb.input import OUTPUT_PREFIX, PQR_FILENAME
from sashimi.tabipb.vtk import read_vtk

__all__ = [
    "DEFAULT_TIMEOUT",
    "LOWEST_RELIABLE_MESH_DENSITY",
    "MIN_ATOMS",
    "TabipbCrash",
    "TabipbRun",
    "run_tabipb",
]

DEFAULT_TIMEOUT = 600.0

# NanoShaper's own limit. It suggests padding with zero-radius dummy atoms;
# sashimi does not do that silently, because a structure that needs inventing
# atoms to be meshable is one the caller should know about.
MIN_ATOMS = 4

# Below this, NanoShaper stops producing a mesh TABI-PB can solve on and the
# run aborts with an uncaught C++ exception. Measured on ALA-GLY: 1.0 fails,
# 1.5 succeeds. Note this is above `BoundaryElementRequest.mesh_density`'s
# default of 1.0 — see `_density_hint`.
LOWEST_RELIABLE_MESH_DENSITY = 1.5

_ENERGY_RE = re.compile(r"Solvation energy\s*=\s*([-+0-9.eE]+)\s*kJ/mol")
_FREE_ENERGY_RE = re.compile(r"Free energy\s*=\s*([-+0-9.eE]+)\s*kJ/mol")

_ERROR_SIGNATURES = (
    "command not found",
    "<<ERROR>>",
    "invalid mesh value",
    "terminating due to uncaught exception",
    "Segmentation fault",
)


class TabipbCrash(SolverCrash):
    """TABI-PB exited abnormally, timed out, or produced no usable output."""


@dataclass
class TabipbRun:
    potential: SurfacePotential | None
    normal_derivative: FloatArray | None
    energy_kj_mol: float | None
    free_energy_kj_mol: float | None
    stdout: str
    wall_seconds: float
    returncode: int


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _density_hint(mesh_density: float) -> str:
    """Name the likeliest cause when a coarse mesh takes the solver down.

    Measured on ALA-GLY: `mesh_density` at or below 1.0 aborts with an uncaught
    C++ exception, 1.5 and above succeeds. The failure carries no clue of its
    own, and the protocol's default is 1.0 — so without this the most likely
    first call a caller makes ends in `terminating due to uncaught exception`.
    """
    if mesh_density >= LOWEST_RELIABLE_MESH_DENSITY:
        return ""
    return (
        f" mesh_density={mesh_density:g} is below {LOWEST_RELIABLE_MESH_DENSITY:g}, which is "
        "where the triangulation stops producing a mesh TABI-PB can solve on; a coarse "
        "mesh is the likeliest cause here. Raise it and try again."
    )


def check_meshable(n_atoms: int) -> None:
    """Refuse a structure the triangulator cannot handle, before running it."""
    if n_atoms < MIN_ATOMS:
        raise InputError(
            f"TABI-PB needs at least {MIN_ATOMS} atoms; this structure has {n_atoms}. "
            "NanoShaper cannot triangulate a surface from fewer, so a single-sphere "
            "case such as the Born ion has no boundary-element answer without adding "
            "dummy atoms — which would change the structure, so sashimi does not do it "
            "for you. Use a finite-difference backend for that case."
        )


def run_tabipb(
    binary: TabipbBinary,
    *,
    pqr_text: str,
    input_text: str,
    n_atoms: int,
    mesh_density: float,
    timeout: float = DEFAULT_TIMEOUT,
    expect_potential: bool = True,
) -> TabipbRun:
    check_meshable(n_atoms)

    with tempfile.TemporaryDirectory(prefix="sashimi-tabipb-") as tmp:
        work = Path(tmp)
        (work / PQR_FILENAME).write_text(pqr_text)
        (work / "tabipb.in").write_text(input_text)

        # See the module docstring: `tabipb` shells out to `NanoShaper` by name.
        env = dict(os.environ)
        env["PATH"] = f"{binary.mesher_path.parent}{os.pathsep}{env.get('PATH', '')}"

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [str(binary.path), "tabipb.in"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                check=False,  # exit code is not the signal; verified structurally below
            )
        except subprocess.TimeoutExpired as exc:
            raise TabipbCrash(
                f"TABI-PB timed out after {timeout:g}s. Triangulation dominates for a "
                "large solute or a high mesh density; lower mesh_density or raise the "
                f"timeout.\nLast output:\n{_tail(_as_text(exc.stdout))}"
            ) from exc
        except OSError as exc:
            raise TabipbCrash(f"could not execute {binary.path}: {exc}") from exc
        wall = time.monotonic() - started

        combined = proc.stdout + proc.stderr
        for signature in _ERROR_SIGNATURES:
            if signature.lower() in combined.lower():
                raise TabipbCrash(
                    f"TABI-PB reported an error ({signature!r}), exit code "
                    f"{proc.returncode}.{_density_hint(mesh_density)}\n"
                    f"Last output:\n{_tail(combined)}"
                )

        potential = None
        normal = None
        vtk_path = work / f"{OUTPUT_PREFIX}.vtk"
        if expect_potential:
            if not vtk_path.is_file():
                produced = sorted(p.name for p in work.iterdir())
                raise TabipbCrash(
                    f"TABI-PB exited {proc.returncode} without writing {vtk_path.name}.\n"
                    f"Files in the working directory: {produced}\n"
                    f"Last output:\n{_tail(combined)}"
                )
            parsed = read_vtk(vtk_path)
            potential = parsed.potential
            normal = parsed.normal_derivative

        energy_match = _ENERGY_RE.search(combined)
        if energy_match is None:
            raise TabipbCrash(
                f"TABI-PB printed no solvation energy.\nLast output:\n{_tail(combined)}"
            )
        free_match = _FREE_ENERGY_RE.search(combined)

        return TabipbRun(
            potential=potential,
            normal_derivative=normal,
            # Already kJ/mol — TABI-PB is the one backend whose *energy* needs
            # no conversion. Its potential is not so kind: that arrives in
            # kJ/mol/e and `backend.to_kt_per_e` divides by RT. Reporting one
            # per mole and the other per mole too is exactly why the field was
            # wrong for six recordings while every energy stayed right.
            energy_kj_mol=float(energy_match.group(1)),
            free_energy_kj_mol=float(free_match.group(1)) if free_match else None,
            stdout=combined,
            wall_seconds=wall,
            returncode=proc.returncode,
        )


def _as_text(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode(errors="replace")
    return raw if isinstance(raw, str) else ""
