"""What unit is TABI-PB's surface potential in? Calibrate against a closed form.

`SurfacePotential.values` is fixed at kT/e by the protocol boundary (ROADMAP
§4), and `sashimi.tabipb` applied no conversion. The corpus said that could not
be right: the same physical potential at 277 K reads 7.64% larger in kT/e than
at 298.15 K, and the two peptide recordings differ by 0.37% -- the screening
response alone. So the values do not carry a 1/T and are not kT/e.

That is a negative. This is the positive: outside a sphere of radius a carrying
q at its centre the potential is exactly q / (4 pi eps0 eps_s r), so every mesh
vertex has a known answer and the ratio to it is the conversion factor. Run at
two temperatures it also re-derives the negative by a second route, and run up a
mesh ladder it separates a fixed unit factor from a shrinking discretization
error -- which matters, because at the default density the ratio misses RT by
1.75%, close enough to be the unit and far enough to be something else.

Reproduces the ROADMAP §12 table "TABI-PB's surface potential is kJ/mol/e".
"""

from __future__ import annotations

import numpy as np

from sashimi.analytic import born_potential
from sashimi.constants import GAS_CONSTANT, JOULES_PER_KJ
from sashimi.protocol import BoundaryElementRequest, PQRData, SolventModel, SurfaceModel
from sashimi.tabipb import TabipbSolver
from sashimi.tabipb.run import TabipbCrash

RADII = (3.0, 4.0)
DENSITIES = (3.0, 5.0, 8.0, 12.0)
TEMPERATURES = (298.15, 277.0)
ATTEMPTS = 4

# NanoShaper refuses fewer than four atoms, so the sphere is built from four
# coincident-to-1e-2-A ones at the vertices of a tiny regular tetrahedron. Their
# union is a sphere to within the offset, and the charge is split equally so the
# tetrahedral symmetry cancels the dipole exactly -- leaving a centred monopole,
# whose exterior potential is the closed form used throughout.
OFFSET = 0.01
TETRAHEDRON = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)


def sphere(radius: float) -> PQRData:
    return PQRData(coords=TETRAHEDRON * OFFSET, charges=np.full(4, 0.25), radii=np.full(4, radius))


def solve(radius: float, density: float, temperature: float):
    """One solve, retried: NanoShaper's handoff to TABI-PB fails intermittently.

    The failure is `stoul: no conversion` from inside the binary, and on this
    input it moved with the thread count rather than with the density, so a
    crash is retried before it is believed.
    """
    solvent = SolventModel(
        surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.0, temperature=temperature
    )
    request = BoundaryElementRequest(
        structure=sphere(radius),
        solvent=solvent,
        mesh_density=density,
        want_potential=True,
    )
    for _ in range(ATTEMPTS):
        try:
            return TabipbSolver().solve(request).potential, solvent
        except TabipbCrash:
            continue
    return None, solvent


def main() -> None:
    rt = GAS_CONSTANT * 298.15 / JOULES_PER_KJ
    print(f"RT at 298.15 K = {rt:.6f} kJ/mol   (the exact kT/e -> kJ/mol/e factor)")
    print()

    print("A: the values do not move with temperature, so they are not kT/e.")
    print(f"{'a':>5} {'T':>8} {'n_vert':>8} {'mean value':>12}")
    for radius in RADII:
        for temperature in TEMPERATURES:
            surf, _ = solve(radius, 3.0, temperature)
            if surf is None:
                print(f"{radius:5.1f} {temperature:8.2f}   CRASHED {ATTEMPTS}/{ATTEMPTS}")
                continue
            print(
                f"{radius:5.1f} {temperature:8.2f} {surf.n_vertices:8d} "
                f"{float(np.mean(surf.values)):12.6f}"
            )
    print("  kT/e would differ by 298.15/277 = 1.07635 between those rows.")
    print()

    print("B: graded per-vertex against the closed form at that vertex's own")
    print("   radius, the ratio converges to RT as the mesh refines.")
    print(
        f"{'a':>5} {'dens':>6} {'n_vert':>8} {'r mean':>8} {'ratio mean':>11} "
        f"{'vs RT':>9} {'excess':>8} {'x dens':>7}"
    )
    for radius in RADII:
        for density in DENSITIES:
            surf, solvent = solve(radius, density, 298.15)
            if surf is None:
                print(f"{radius:5.1f} {density:6.1f}   CRASHED {ATTEMPTS}/{ATTEMPTS}")
                continue
            r = np.linalg.norm(surf.vertices, axis=1)
            exact = np.array(
                [born_potential(ri, 1.0, solvent.solvent_dielectric, 298.15) for ri in r]
            )
            ratio = float(np.mean(surf.values / exact))
            excess = ratio / rt - 1.0
            print(
                f"{radius:5.1f} {density:6.1f} {surf.n_vertices:8d} {r.mean():8.4f} "
                f"{ratio:11.5f} {ratio / rt:9.5f} {excess * 100:7.3f}% "
                f"{excess * 100 * density:7.3f}"
            )
    print()
    print("  'excess' is how far above RT the ratio sits; 'x dens' is that times")
    print("  the density. It is constant down each radius, so the excess is pure")
    print("  h^2 discretization and extrapolates to zero: the factor is exactly RT.")


if __name__ == "__main__":
    main()
