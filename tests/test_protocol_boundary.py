"""The protocol seam, guarded.

ROADMAP.md §10 defers extracting `pb-protocol` until debye exists, on the
condition that the seam cannot erode in the meantime. This is that condition:
`{protocol, dx, pqr, errors}` must import nothing else from sashimi, so the
extraction stays mechanical whenever it happens.

It also enforces the layering rule CLAUDE.md states — no APBS vocabulary above
`sashimi.apbs` — which is a stronger claim than "the imports are clean", and is
the one that actually keeps the protocol solver-neutral.
"""

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "sashimi"

# The closed set that would become pb-protocol. Nothing here may import
# anything from sashimi outside this set.
PROTOCOL_MODULES = ("protocol.py", "dx.py", "pqr.py", "errors.py")
PROTOCOL_NAMES = {"sashimi." + m.removesuffix(".py") for m in PROTOCOL_MODULES}

# Third-party dependencies the extracted package would be allowed to carry,
# on top of the standard library. fastmcp, pydantic and pdb2pqr must never
# appear here — debye's portability claim is undercut if the shared types drag
# an MCP server along.
ALLOWED_THIRD_PARTY = {"numpy"}

# APBS's input vocabulary. None of it belongs above sashimi.apbs — neither the
# keyword names nor, just as importantly, their values.
APBS_VOCABULARY = (
    "mg-auto",
    "mg-manual",
    "cglen",
    "fglen",
    "dime",
    "chgm",
    "srfm",
    "bcfl",
    "smol",
    "spl2",
    "spl4",
    "sdh",
    "mdh",
)

# Phase 4 paid the debt this used to carry: `SolventModel.surface_method`, once
# Literal["smol", "spl2", "mol"], is now a solver-neutral `SurfaceModel` enum
# and the APBS keywords live in `sashimi.apbs.options`. The strict xfail that
# tracked it is gone because it started passing, which is what strict is for.
VOCABULARY_MODULES = PROTOCOL_MODULES


# The backends debye must not reach into. ROADMAP.md sections 10 and 12 both
# say this file is what holds that line — it did not, until debye existed to be
# held: every test above is parametrized over the four protocol modules alone,
# so the claim was about a module nothing checked.
SOLVER_PACKAGES = ("apbs", "delphi", "tabipb", "gb")


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


@pytest.mark.parametrize("module", PROTOCOL_MODULES)
def test_protocol_modules_form_a_closed_set(module):
    """Extraction must stay mechanical: no edges leaving the set."""
    internal = {i for i in imports_of(SRC / module) if i.startswith("sashimi")}
    escaping = {i for i in internal if i not in PROTOCOL_NAMES}
    assert not escaping, (
        f"{module} imports {sorted(escaping)}, which is outside the pb-protocol set "
        f"{sorted(PROTOCOL_NAMES)}. Either the import belongs inside the set, or the "
        "seam described in ROADMAP.md section 10 has eroded."
    )


@pytest.mark.parametrize("module", PROTOCOL_MODULES)
def test_protocol_modules_stay_dependency_light(module):
    """debye's pitch is that it runs anywhere; shared types must not weigh it down."""
    roots = {i.split(".")[0] for i in imports_of(SRC / module) if not i.startswith("sashimi")}
    heavy = roots - ALLOWED_THIRD_PARTY - sys.stdlib_module_names
    assert not heavy, (
        f"{module} imports {sorted(heavy)}. The extractable protocol set carries numpy "
        "and the standard library only — see ROADMAP.md section 10."
    )


@pytest.mark.parametrize("module", VOCABULARY_MODULES)
def test_no_apbs_vocabulary_above_the_backend(module):
    """CLAUDE.md's layering rule, enforced rather than trusted.

    `SolventModel.surface_method` carried APBS's `srfm` values through phases
    1-3. Phase 4 replaced it with a solver-neutral `SurfaceModel`; this test is
    what stops the leak recurring.
    """
    text = (SRC / module).read_text().lower()
    # Comments may discuss APBS; code may not use its vocabulary as values.
    code = "\n".join(
        line.split("#")[0] for line in text.splitlines() if not line.strip().startswith("#")
    )
    leaked = [term for term in APBS_VOCABULARY if f'"{term}"' in code or f"'{term}'" in code]
    assert not leaked, (
        f"{module} uses APBS vocabulary {leaked} as literal values. Everything above "
        "sashimi.apbs speaks physics, not APBS input keywords."
    )


@pytest.mark.parametrize("module", sorted(p.name for p in (SRC / "debye").glob("*.py")))
def test_debye_does_not_reach_into_another_backend(module):
    """The clean-room claim, enforced where it is cheapest to keep.

    debye is in-repo (ROADMAP.md section 12), so `from sashimi.apbs.grid import
    size_grid` would work, would look like reuse, and would quietly make the
    reference solver a dependency of the solver being validated against it.
    What debye may share is physics — `sashimi.analytic`, `sashimi.constants`,
    the protocol set — and not another solver's arithmetic.
    """
    internal = {i for i in imports_of(SRC / "debye" / module) if i.startswith("sashimi")}
    reached = {i for i in internal if i.split(".")[1] in SOLVER_PACKAGES}
    assert not reached, (
        f"debye/{module} imports {sorted(reached)}. debye is a clean-room solver: it may "
        "share the protocol and the physics, never another backend's implementation."
    )


@pytest.mark.parametrize("module", sorted(p.name for p in (SRC / "debye").glob("*.py")))
def test_debye_carries_no_apbs_vocabulary(module):
    """Same rule as the protocol set, for the same reason, one layer down."""
    text = (SRC / "debye" / module).read_text().lower()
    code = "\n".join(
        line.split("#")[0] for line in text.splitlines() if not line.strip().startswith("#")
    )
    leaked = [term for term in APBS_VOCABULARY if f'"{term}"' in code or f"'{term}'" in code]
    assert not leaked, (
        f"debye/{module} uses APBS vocabulary {leaked} as literal values. debye has its "
        "own lattice and its own boundary condition; naming them after APBS's keywords "
        "is how two solvers come to be assumed identical."
    )
