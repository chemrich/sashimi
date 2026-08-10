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
