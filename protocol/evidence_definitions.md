# Evidence Definitions

## E0 — Source-level evidence
Contains:
- problem statement/specification
- input constraints
- required output
- candidate source code

Does not contain:
- paired counterpart implementation
- hidden intervention label
- pair-relative speedup
- future evidence

## E1 — Structured self-diagnosis
Generated independently by each evaluated model.

A model's E1 must not be reused as another model's self-diagnosis.

## E2 — Static program evidence
Deterministic structural facts derived without running the program, including:
- executable LOC
- AST node count
- function/recursion counts
- loops and maximum loop nesting
- branch/comprehension/sort counts
- calls inside loops
- allocation/import counts

E2 is neutral and contains no efficiency label or recommendation.

## E3 — Dynamic profiling evidence
Controlled absolute observations including:
- runtime
- memory
- correctness-audit execution counts
- input-size summaries

E3 does not reveal:
- counterpart runtime
- relative speedup
- pair role
- final OPTIMIZE/PRESERVE label
