"""Test package.

A package rather than loose modules so mypy can address it as `tests.*` — its
override patterns only accept whole-component wildcards, so `test_*` is not
expressible.
"""
