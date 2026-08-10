"""Naming and lifetime of the files a solve leaves behind.

A potential map is large — 12 MB at 97³, 56 MB at the `max_points` cap — and
tools return a *path* to one rather than its contents, because inlining a grid
would cost millions of tokens (ROADMAP.md §6). That makes two things the
caller's problem, and this module makes them explicit rather than implicit.

**Naming.** A map is addressed by the content that produced it: the structure
plus the resolved parameter set. Two solves that differ in any way that changes
the answer get different filenames, so re-solving cannot silently overwrite a
previous map, and a map on disk can be matched back to the request that made
it. Two solves that are genuinely identical share a file, which is a cache for
free.

**Lifetime.** sashimi never deletes a map it has written. Maps accumulate until
something else removes them, and at these sizes that matters within a single
busy session. The contract is stated here and in the tool descriptions so it is
a decision rather than a surprise; `describe_cleanup` is what tools quote.

**Locality.** Paths assume the caller shares a filesystem with the server. That
holds for the stdio transport sashimi ships and fails for anything remote,
which is what MCP Resources would fix when a remote transport exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sashimi.protocol import PQRData

__all__ = ["ADDRESS_LENGTH", "content_address", "describe_cleanup", "map_path"]

ADDRESS_LENGTH = 12  # hex characters; 48 bits, ample against accidental collision

CLEANUP_CONTRACT = (
    "sashimi does not delete potential maps it writes. Filenames are content-"
    "addressed, so re-solving the same structure with the same parameters "
    "reuses a file rather than growing the directory, but distinct solves "
    "accumulate — a 97^3 map is ~12 MB. Remove them when done."
)

LOCALITY_CONTRACT = (
    "Paths are local to the machine running the server. This is a stdio server, "
    "so that is normally the same machine as the client."
)


def content_address(structure: PQRData, resolved_parameters: dict[str, Any]) -> str:
    """A short, stable digest of everything that determines the answer.

    Covers coordinates, charges and radii — not atom labels, which do not reach
    the solver — together with the backend's *resolved* parameters, so a grid
    that was relaxed or a surface model that was mapped changes the address.
    """
    digest = hashlib.sha256()
    for array in (structure.coords, structure.charges, structure.radii):
        digest.update(np_bytes(array))
    # sort_keys so dict ordering cannot change the address; default=str so an
    # unexpected value type degrades to something stable rather than raising.
    digest.update(json.dumps(resolved_parameters, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:ADDRESS_LENGTH]


def np_bytes(array: Any) -> bytes:
    """Canonical bytes for an array, independent of memory layout."""
    return bytes(memoryview(array.astype("<f8", copy=False).tobytes()))


def map_path(
    base: str | os.PathLike[str],
    address: str,
    *,
    stem: str = "potential",
    suffix: str = ".dx",
) -> Path:
    """Where a map with this address lives, alongside `base`."""
    directory = Path(base).expanduser().resolve().parent
    return directory / f"{stem}-{address}{suffix}"


def describe_cleanup() -> str:
    """One sentence for tool descriptions and documentation."""
    return f"{CLEANUP_CONTRACT} {LOCALITY_CONTRACT}"
