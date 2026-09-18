# Dataset

## Files

### `benchmark_instances.csv`
The 120 public evaluation instances. Each row uses a neutral `Instance ID` and contains the problem text, constraints, required output, candidate source code, and original problem URL.

**No OPTIMIZE/PRESERVE label is included in this file.**

### `pair_validation.csv`
Pair-level construction evidence used to validate the selected efficient/inefficient pair relationship. This file is for provenance and audit; pair-relative information was not exposed to evaluated models.

### `e0_prompts.csv`
Frozen source-level prompts used at E0.

### `e2_static_evidence.csv`
Deterministic Python static-analysis facts such as LOC, AST node count, loop counts, nesting, branches, calls inside loops, allocations, and imports.

### `e3_dynamic_evidence.csv`
Absolute runtime/memory and correctness-audit information. The neutral E3 text intentionally excludes counterpart timing, relative speedup, hidden ACEOB role, and final intervention label.

### `ground_truth.csv`
Final hidden role and OPTIMIZE/PRESERVE label, released for post-evaluation reproducibility.

**Never use this file as model-visible input when reproducing the reported evaluation.**

### `benchmark_summary.csv`
Compact frozen-benchmark statistics.

## Canonical join key

Use `Instance ID` to join:
- `benchmark_instances.csv`
- `e0_prompts.csv`
- `e2_static_evidence.csv`
- `e3_dynamic_evidence.csv`
- `ground_truth.csv`

The expected canonical ID set is:

`ADSO_TEST_001` through `ADSO_TEST_120`.

## Label interpretation

- `OPTIMIZE`: benchmark evidence indicates that performance intervention is warranted.
- `PRESERVE`: intervention is not warranted relative to the benchmark's validated efficiency evidence.

`PRESERVE` should **not** be interpreted as proof of global program optimality.
