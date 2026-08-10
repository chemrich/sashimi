"""Run the README's Python examples.

The README's headline example was broken for four merges: phase 4 changed the
`Solver` signature and the docs were not updated with it. Nothing caught it,
because the suite was thorough about code and silent about prose. This is the
missing check.

Blocks execute in a temp directory containing `protein.pqr`, since that is the
filename the README uses. A block that cannot run standalone can opt out with a
`# test: skip` comment, but the default is that documented code is executed
code.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
FIXTURE_PQR = Path(__file__).resolve().parent / "data" / "ala-gly.pqr"

PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
SKIP_DIRECTIVE = "# test: skip"

# A block that builds a solver shells out to APBS; one that only imports does
# not. Marking per block keeps the import-only examples in the binary-free tier.
BINARY_MARKERS = ("ApbsSolver", "StubBemSolver", "corpus")


def extract_blocks(markdown: str) -> list[str]:
    return [match.group(1) for match in PYTHON_BLOCK.finditer(markdown)]


BLOCKS = extract_blocks(README.read_text())


def _skip_marks(code: str) -> list[pytest.MarkDecorator]:
    if SKIP_DIRECTIVE in code:
        return [pytest.mark.skip(reason="opted out with a # test: skip directive")]
    return []


# Two param lists rather than one, because the marks differ per test. Sharing a
# single list would put `apbs` on the syntax check too, and a syntax check that
# needs a solver binary installed is not a syntax check.
RUN_CASES = [
    pytest.param(
        code,
        id=f"block-{index}",
        marks=(
            _skip_marks(code)
            or ([pytest.mark.apbs] if any(m in code for m in BINARY_MARKERS) else [])
        ),
    )
    for index, code in enumerate(BLOCKS)
]

SYNTAX_CASES = [
    pytest.param(code, id=f"block-{index}", marks=_skip_marks(code))
    for index, code in enumerate(BLOCKS)
]


def test_the_extractor_finds_something() -> None:
    """A regex that silently matches nothing would make this file vacuous.

    This is the guard that matters most here: every other test in the module is
    parametrized over `BLOCKS`, so an empty list means the whole file passes
    while checking nothing.
    """
    assert BLOCKS, f"no ```python blocks found in {README.name} — has the fence style changed?"


@pytest.mark.parametrize("code", RUN_CASES)
def test_readme_example_runs(code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute a documented example exactly as written."""
    shutil.copy(FIXTURE_PQR, tmp_path / "protein.pqr")
    monkeypatch.chdir(tmp_path)

    try:
        exec(compile(code, f"{README.name} block", "exec"), {"__name__": "__readme__"})
    except Exception as exc:
        pytest.fail(
            f"a README example failed: {type(exc).__name__}: {exc}\n\n"
            f"The documented code is:\n{code}"
        )


@pytest.mark.parametrize("code", SYNTAX_CASES)
def test_readme_example_is_syntactically_valid(code: str) -> None:
    """Cheap and binary-free: catches a typo even when APBS is absent."""
    compile(code, f"{README.name} block", "exec")
