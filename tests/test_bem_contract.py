"""Phase 4's exit criterion: does the protocol admit a BEM backend?

Not "could it, in principle" — an actual `Solver` implementation returning
surface potentials, exercised through the same types APBS uses. If any of these
tests needs an APBS-shaped concession to pass, the protocol is still wrong.

No `apbs` marker: the stub computes analytically, so this whole file runs in
the binary-free tier. That is itself part of the point — the protocol layer is
testable everywhere, which is what ROADMAP.md §7 promises and what debye's
portability claim depends on.
"""

import numpy as np
import pytest

from sashimi.bem_stub import StubBemSolver
from sashimi.errors import InputError, SolverError, UnsupportedRequest
from sashimi.protocol import (
    BoundaryElementRequest,
    Equation,
    FiniteDifferenceRequest,
    PotentialGrid,
    PQRData,
    SolventModel,
    SurfaceModel,
    SurfacePotential,
)
from tests.helpers import surface


@pytest.fixture
def ion():
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([3.0]),
        labels=("ION 1 I",),
    )


class TestRequestShape:
    def test_bem_requests_cannot_express_a_grid(self, ion):
        """The FD/BEM split is structural, not a runtime check."""
        request = BoundaryElementRequest(structure=ion)
        assert not hasattr(request, "grid")

    def test_bem_requests_cannot_express_an_equation(self, ion):
        """A nonlinear BEM request is unrepresentable, not merely rejected —
        BEM is built on the linearized operator's Green function."""
        request = BoundaryElementRequest(structure=ion)
        assert not hasattr(request, "equation")

    def test_fd_requests_carry_both(self, ion):
        request = FiniteDifferenceRequest(structure=ion, equation=Equation.NONLINEAR)
        assert request.grid.resolution > 0
        assert request.equation is Equation.NONLINEAR

    def test_both_families_share_the_solvent_model(self, ion):
        """Solvent is physics, so it belongs to every family."""
        solvent = SolventModel(ionic_strength=0.2)
        assert BoundaryElementRequest(structure=ion, solvent=solvent).solvent is solvent
        assert FiniteDifferenceRequest(structure=ion, solvent=solvent).solvent is solvent

    def test_a_request_for_nothing_is_rejected(self, ion):
        with pytest.raises(ValueError, match="wants neither energy nor potential"):
            BoundaryElementRequest(structure=ion, want_energy=False, want_potential=False)


class TestResultShape:
    def test_a_bem_backend_returns_a_surface_potential(self, ion):
        result = StubBemSolver().solve(BoundaryElementRequest(structure=ion))
        mesh = surface(result)
        assert mesh.n_vertices > 0
        assert mesh.vertices.shape == (mesh.n_vertices, 3)
        assert mesh.stats()["kind"] == "surface"

    def test_energy_is_the_universal_currency(self, ion):
        """Both families produce it; only one produces a volume."""
        bem = StubBemSolver().solve(BoundaryElementRequest(structure=ion))
        assert bem.energy_kj_mol is not None
        assert bem.energy_kj_mol < 0, "solvating a charge releases energy"

    def test_potential_can_be_declined(self, ion):
        """A backend must be allowed to return energy only."""
        result = StubBemSolver().solve(BoundaryElementRequest(structure=ion, want_potential=False))
        assert result.potential is None
        assert result.energy_kj_mol is not None

    def test_the_result_type_is_shared_across_families(self, ion):
        """One SolveResult, two potential shapes — no per-family result type."""
        bem = StubBemSolver().solve(BoundaryElementRequest(structure=ion))
        assert isinstance(bem.potential, SurfacePotential)
        assert not isinstance(bem.potential, PotentialGrid)
        # Same attributes a caller uses on an APBS result.
        assert bem.provenance.backend
        assert isinstance(bem.diagnostics, dict)

    def test_a_backend_must_deliver_what_was_asked(self, ion):
        """check_satisfies is the contract, not a suggestion."""
        result = StubBemSolver().solve(BoundaryElementRequest(structure=ion))
        result.potential = None
        with pytest.raises(ValueError, match="asked for potential"):
            result.check_satisfies(BoundaryElementRequest(structure=ion))


class TestUnsupportedIsAnInputError:
    def test_a_bem_backend_refuses_a_gaussian_dielectric(self, ion):
        """BEM needs a sharp interface; refusing is not a crash."""
        request = BoundaryElementRequest(
            structure=ion, solvent=SolventModel(surface_model=SurfaceModel.GAUSSIAN)
        )
        with pytest.raises(UnsupportedRequest, match="sharp dielectric interface"):
            StubBemSolver().solve(request)

    def test_unsupported_is_an_input_error_not_a_solver_error(self):
        """The caller fixes the request; nothing was wrong with the solver."""
        assert issubclass(UnsupportedRequest, InputError)
        assert not issubclass(UnsupportedRequest, SolverError)


class TestProvenanceIsUniversal:
    def test_every_backend_reports_provenance(self, ion):
        # An explicit surface model, because this asserted the protocol default
        # and so re-encoded it: the echo has to be the *request's* value, which
        # a test agreeing with whatever the dataclass says cannot show.
        request = BoundaryElementRequest(
            structure=ion,
            mesh_density=0.5,
            solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        )
        result = StubBemSolver().solve(request)
        assert result.provenance.backend == "stub-bem-0"
        # Resolved parameters record what the backend actually used.
        assert result.provenance.resolved_parameters["mesh_density"] == 0.5
        assert result.provenance.resolved_parameters["surface_model"] == "van-der-waals"

    def test_mesh_density_changes_the_mesh(self, ion):
        coarse = StubBemSolver().solve(BoundaryElementRequest(structure=ion, mesh_density=0.2))
        fine = StubBemSolver().solve(BoundaryElementRequest(structure=ion, mesh_density=2.0))
        assert surface(fine).n_vertices > surface(coarse).n_vertices
