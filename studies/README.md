# `studies/` — the generators behind ROADMAP.md section 12's tables

Every number in §12 that is not a corpus recording came from a script. Until
2026-08-26 those scripts lived in a scratch directory and several were already
gone, which put §12 in the position it criticises other people's numbers for:
**a figure whose provenance is not recoverable from anything checked in.** §12
says exactly that about the 24-window `converging` generator and about the width
sweep's 4.6–9.0×, both of which are still unreproducible. This directory exists
so the next one is not.

## The rule

**Results belong in `tests/corpus/`. Generators belong here.** The corpus is
tiered, versioned, and a diff to it is treated as a real result change; that is
the mechanism that stops a *number* rotting. This directory stops the *method*
rotting, which is a different problem and needs a different answer.

A script here is a record of how a table was produced, not library code. It
prints, it hard-codes the configuration its table was taken at, and nothing
imports it. `pyproject.toml` says so in `per-file-ignores` and gives the reason.

## What runs on every push

`tests/test_studies.py` runs the cheapest study end to end and checks it
reproduces the output checked in beside it. That is the whole guard, and it is
deliberately one test: the alternative — running the campaign — is tens of
minutes, and "too slow to check here" is one step from "quietly absent", which
is the failure §7 records the DelPhi tier making for a year while CI stayed
green. Everything else here is run by hand.

**If that test goes red, the question is not "fix the test".** It is whether
§12's tables moved, and if they did they need re-taking and the document needs
correcting.

## How they were lost, because the lesson is not the obvious one

Four generators went missing between being run and being written up, and it was
not a `/tmp` reboot. They were written *inside* a `git worktree` created to
measure a branch, and `git worktree remove --force` took them with it. Their
outputs, which had been redirected to the parent directory, all survived.

**A worktree is a working copy, not a workspace.** Anything you want to keep
goes outside it. All four have been reconstructed from their surviving outputs
and their siblings, and each reproduces its recorded table bit-identically —
which is stated per-file and is the only thing that makes "reconstructed"
mean anything.

## The map

Run everything from the repository root.

### `field_axis/` — ROADMAP §12, "The field axis, measured"

| script | what it produced |
|---|---|
| `field_sphere.py` | Arm A: worst-direction error at a fixed physical radius, 21 paddings, both radii |
| `sphere_shell_phase.py` | Arm A by Arm B's statistic — the table showing the two summaries disagree *(reconstructed)* |
| `sphere_shell.py` | the single-lattice version that showed the sweep was needed *(reconstructed)* |
| `field_real.py` | Arm B: the `fas2` table at both coarse resolutions, four referees |
| `field_phase.py` | Arm B: the five-padding `ala-gly` table *(reconstructed)* |
| `refspread.py` | the referee's own behaviour — same-scheme refinement against cross-scheme spread |
| `refladder.py` | the referee ladder, `h` = 0.25 / 0.15 / 0.12 / 0.10 / 0.09 |
| `field_nested.py` | the exactly-nested control: index-read against `value_at`, five paddings |
| `isolate.py` | interpolation separated from referee resolution, two paddings |
| `cheap_ref.py` | the referee choice behind the shipped test's bar |
| `energy_ladder_mol.py` | "the two axes disagree on one fixture" — the four-rung energy ladder *(reconstructed)* |
| `quantize.py` | **the result that did not reproduce**, kept for that reason *(reconstructed)* |
| `shared_limit.py` | δ(h) between the two schemes on one lattice, with two null controls — the referee-free half of the shared-limit test |
| `consistency_born.py` | each scheme against the *exact* Born field — the half that settles it |

### `union_gap/` — ROADMAP §12, "The other half of that bug"

| script | what it produced |
|---|---|
| `exact_union_sdf.py` | two independent exact signed distances to a union of balls, used to referee the repair |
| `two_ball_oracle.py` | the two-ball closed form, now also in `tests/test_debye_m4.py` as `_two_ball_gap` |
| `vdw_gap_fix.py` | the prototype repair, superseded by `ReducedSurface._union_signed_gap` |
| `band_audit.py` | "19.0% of the interior band faces" |
| `energy_effect.py` | the `fas2` energy move, +44.79 kJ/mol at 1.0 Å and +10.06 at 0.5 Å |
| `ladder.py` | the five-rung `ala-gly` van der Waals ladder |
| `born_ramp.py` | the twelve-configuration Born bit-identity check |
| `bench_vdw.py`, `bench_mol.py` | the cost ratios, min-of-3 and interleaved |

### `refinement/` — ROADMAP §12, "The noise floor `converging` does not need"

| script | what it produced |
|---|---|
| `step_vs_pose.py` | the last Richardson step against the pose dispersion at the matching rung |

### `tabipb_units/` — ROADMAP §12, "TABI-PB's surface potential was kJ/mol/e, labelled kT/e"

| script | what it produced |
|---|---|
| `born_sphere.py` | both halves: the values do not move with temperature, and the ratio to the closed form converges to RT down a four-rung mesh ladder at two radii |

Needs TABI-PB and NanoShaper installed. NanoShaper's handoff fails
intermittently on this input (`stoul: no conversion`) and the script retries
before believing a crash; `sdens` below 1.5 fails for real and is not swept.

## What is not here

The scripts behind the *adversarial* passes — the ones that refuted claims
rather than produced tables — are not checked in. They verified numbers rather
than generating them, and every finding they carried has since been written into
§12 or a docstring with its own evidence. If one is needed again it is cheaper
to rewrite than to maintain.
