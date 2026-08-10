"""Failure modes, split by who can act on them.

Three branches, because a caller does three different things with them:

- `InputError` — the request cannot be satisfied as asked. Fix the request.
- `SolverError` — the request was reasonable; the solver failed on it. Retry
  with different parameters, or believe the solver.
- `BackendUnavailable` — nothing was even attempted. Fix the installation.

The previous split (backend-neutral vs APBS-specific) told a caller which
module raised, which is not a distinction anyone acts on. Backend-specific
subclasses still exist — `ApbsNotFound`, `ApbsCrash` — but they now hang off
the branch that describes the *kind* of failure rather than forming their own
axis. See ROADMAP.md §4.2.
"""

from __future__ import annotations

__all__ = [
    "BackendUnavailable",
    "ConvergenceFailure",
    "GridTooLarge",
    "InputError",
    "MalformedStructure",
    "PreparationFailed",
    "SashimiError",
    "SolverCrash",
    "SolverError",
    "UnsupportedRequest",
]


class SashimiError(Exception):
    """Base for every error sashimi raises deliberately."""


# --- input ------------------------------------------------------------------


class InputError(SashimiError):
    """The request cannot be satisfied as asked. The caller should change it."""


class MalformedStructure(InputError, ValueError):
    """A structure file could not be parsed, or describes no atoms.

    Also a `ValueError`, because that is what a parser raising on bad input
    idiomatically is, and callers already catch it that way.
    """


class GridTooLarge(InputError):
    """The requested resolution and padding cannot fit within `max_points`."""


class UnsupportedRequest(InputError):
    """This backend cannot honor part of the request.

    Distinct from a crash: the solver is healthy and the request is well formed,
    but they are incompatible — a nonlinear equation asked of a linear-only
    backend, or a surface model the backend has no equivalent for. The message
    names what was asked and what the backend supports.
    """


# --- solver -----------------------------------------------------------------


class SolverError(SashimiError):
    """The solver ran, or tried to, and did not produce a usable answer."""


class ConvergenceFailure(SolverError):
    """The solver ran but did not reach a usable solution."""


class SolverCrash(SolverError):
    """The solver exited abnormally, timed out, or produced no usable output."""


class PreparationFailed(SolverError):
    """Structure preparation (pdb2pqr) could not produce a usable PQR."""


# --- environment ------------------------------------------------------------


class BackendUnavailable(SashimiError):
    """No usable backend could be located or executed.

    Named for the condition rather than the category, because `EnvironmentError`
    is a built-in alias for `OSError` and shadowing it would be worse than a
    slightly off-axis name.
    """
