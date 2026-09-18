# Provenance

## Upstream source

The final held-out ADSO benchmark is derived from the **ACEOB test split**.

ACEOB repository:
https://github.com/CodeGeneration2/ACEOB

ACEOB paper DOI:
https://doi.org/10.1016/j.jss.2024.112250

## Final ADSO benchmark lineage

1. Start from ACEOB test-split material.
2. Screen candidate efficient/inefficient relationships.
3. Validate functional correctness and pair suitability.
4. Exclude ambiguous Level-C cases.
5. Construct a 60-problem / 120-program balanced benchmark.
6. Randomize programs with seed `20260915`.
7. Replace source pair identity with neutral IDs `ADSO_TEST_001` ... `ADSO_TEST_120`.
8. Freeze the benchmark before final evaluated-LLM runs.
9. Keep private ground truth separate from all model-visible evidence.

## Development artifacts intentionally not included in the public dataset folder

The working project also contains:
- candidate-screening workbooks
- semantic-triage workbooks
- earlier validated benchmark versions
- the 8-pair / 16-instance development benchmark
- policy-development workbooks

These files are development history, not part of the reported 120-instance held-out test set.

## Ground-truth strength levels

The frozen workbook records:
- 41 Level-A pairs
- 19 Level-B pairs
- Level-C cases excluded

The released `ground_truth.csv` retains the recorded strength field for auditability.

## Redistribution note

This repository must clearly attribute ACEOB and retain original problem URLs. Before public release, repository owners should confirm that any redistributed third-party problem text/source code is compatible with upstream and platform terms.
