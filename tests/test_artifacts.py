"""Content addressing and the artifact contract.

Binary-free: addressing is a pure function of the request, which is the point —
you can tell whether two solves would collide without running either.
"""

import numpy as np

from sashimi.artifacts import ADDRESS_LENGTH, content_address, describe_cleanup, map_path
from sashimi.protocol import PQRData


def structure(charge: float = 1.0, radius: float = 3.0) -> PQRData:
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([charge]),
        radii=np.array([radius]),
    )


PARAMS = {"surface_model": "smoothed-molecular", "ionic_strength": 0.15}


class TestAddressing:
    def test_is_stable_across_calls(self):
        assert content_address(structure(), PARAMS) == content_address(structure(), PARAMS)

    def test_is_short_and_hex(self):
        address = content_address(structure(), PARAMS)
        assert len(address) == ADDRESS_LENGTH
        int(address, 16)  # raises if not hex

    def test_key_order_does_not_change_it(self):
        """Otherwise a dict rebuild would look like a different calculation."""
        reordered = dict(reversed(list(PARAMS.items())))
        assert content_address(structure(), reordered) == content_address(structure(), PARAMS)

    def test_different_parameters_give_different_addresses(self):
        other = {**PARAMS, "ionic_strength": 0.30}
        assert content_address(structure(), other) != content_address(structure(), PARAMS)

    def test_surface_model_changes_the_address(self):
        """The largest modelling confounder must never share a filename."""
        other = {**PARAMS, "surface_model": "molecular"}
        assert content_address(structure(), other) != content_address(structure(), PARAMS)

    def test_different_structures_give_different_addresses(self):
        assert content_address(structure(charge=-1.0), PARAMS) != content_address(
            structure(), PARAMS
        )
        assert content_address(structure(radius=2.0), PARAMS) != content_address(
            structure(), PARAMS
        )

    def test_labels_do_not_change_the_address(self):
        """Atom names never reach the solver, so they cannot change the answer."""
        labelled = PQRData(
            coords=np.zeros((1, 3)),
            charges=np.array([1.0]),
            radii=np.array([3.0]),
            labels=("ION 1 I",),
        )
        assert content_address(labelled, PARAMS) == content_address(structure(), PARAMS)


class TestMapPath:
    def test_sits_beside_the_source(self, tmp_path):
        source = tmp_path / "structure.pqr"
        assert map_path(source, "abc123").parent == tmp_path

    def test_embeds_the_address(self, tmp_path):
        path = map_path(tmp_path / "s.pqr", "deadbeef1234")
        assert path.name == "potential-deadbeef1234.dx"

    def test_distinct_addresses_never_collide(self, tmp_path):
        a = map_path(tmp_path / "s.pqr", content_address(structure(), PARAMS))
        b = map_path(tmp_path / "s.pqr", content_address(structure(charge=2.0), PARAMS))
        assert a != b


def test_the_cleanup_contract_is_stated_not_implied():
    text = describe_cleanup()
    assert "does not delete" in text
    assert "local to the machine" in text
