"""Markers that skip, rather than merely select.

`@pytest.mark.apbs` and its siblings choose which tests `-m` runs. They have
never *skipped* anything: a marked test on a machine without the binary runs and
fails. CI hid that by deselecting the markers instead, so "sashimi works with
nothing installed" — the property `sashimi.gb` exists to provide, and the one
protean's fallback path depends on — was never tested at all. A bare checkout
failed 56 tests.

Four bugs of this exact shape surfaced on 2026-08-12 alone: a boundary-element
MCP test that ran wherever TABI-PB was absent, a cross-validation suite that
read its precondition off a function counting an always-available backend, a
guard asserted on a flavour that cannot produce it, and a measurement step that
named cases outside its tier. Each looked like coverage and was silence. Fixing
them one at a time was treating instances of a rule that was never enforced.

So the rule is enforced here, once, for every marked test present and future.
`installed_or_skip` keeps the distinction those fixtures were careful about:
absent is normal and skipping is right, but being *pointed at* a binary that
then fails to discover is a broken installation, and skipping there would report
the same green as a working one.

What stops this from hiding real breakage is not this file: it is CI's "Verify
the <backend> tier actually ran" steps, which assert each tier ran wherever its
binary exists. A skip is only safe when something else insists the tests are not
always skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from sashimi.apbs.discover import discover_apbs
from sashimi.delphi.discover import discover_delphi
from sashimi.tabipb.discover import discover_tabipb
from tests.helpers import installed_or_skip

# Marker name -> how to find that backend, and the variable that makes a failure
# to find it an error rather than an absence.
REQUIRED_BINARIES: dict[str, tuple[Callable[[], Any], str]] = {
    "apbs": (discover_apbs, "SASHIMI_APBS_PATH"),
    "delphi": (discover_delphi, "SASHIMI_DELPHI_PATH"),
    "tabipb": (discover_tabipb, "SASHIMI_TABIPB_PATH"),
}


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a marked test when its backend is not installed."""
    for marker, (discover, env_var) in REQUIRED_BINARIES.items():
        if marker in item.keywords:
            installed_or_skip(discover, env_var)
