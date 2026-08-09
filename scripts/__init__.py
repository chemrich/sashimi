"""Maintenance scripts.

A package so that `scripts.build_corpus` resolves to one module name — mypy
rejects a file reachable as both `build_corpus` and `scripts.build_corpus`.
"""
