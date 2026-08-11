"""Generalized Born — the fast approximation, and the first in-process backend.

ROADMAP.md section 8's triage tier: seconds instead of a minute, for deciding
which of a hundred structures deserves a Poisson-Boltzmann solve. It is an
approximation and says so in provenance (`AccuracyTier.APPROXIMATE`), which is
what lets `sashimi validate` report its distance from the reference solvers as a
measurement rather than as a failure.

There is nothing to install. No binary, no environment variable, no discovery,
no CI build step — so unlike every other backend here, this tier cannot silently
skip.
"""

from __future__ import annotations

from sashimi.gb.backend import BACKEND_VERSION, GbSolver
from sashimi.gb.options import GbModel, GbOptions

__all__ = ["BACKEND_VERSION", "GbModel", "GbOptions", "GbSolver"]
