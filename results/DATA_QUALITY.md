# Model Output Data Quality

This document explains exactly how the public result files were cleaned.

## Canonical held-out ID set

Only `ADSO_TEST_001` through `ADSO_TEST_120` are part of the final held-out benchmark.

## GPT-OSS

### C0-C3
- Raw result rows: 487
- Canonical held-out rows retained: 480
- Non-canonical rows excluded: 7
- Expected canonical structure: 120 instances × 4 conditions = 480 rows
- Canonical status: Counter({'SUCCESS': 480})

The excluded rows use development IDs (`ADSO_V1_*`) and are not part of the frozen held-out benchmark.

### E1
- Raw rows: 125
- Final canonical rows: 120
- Technical/API retry rows moved to `results/gpt_oss/retry_audit/`: 5
- Final canonical status: Counter({'SUCCESS': 120})

The API-error rows are preserved for audit but are **not counted as independent scientific observations**.

### Ablations
- Rows: 360
- SUCCESS: 355
- MALFORMED: 5

Malformed outputs are retained rather than silently repaired or regenerated.

## Llama 3.1 8B

### C0-C3
- Rows: 480
- Status: Counter({'SUCCESS': 480})

### E1
- Rows: 120
- SUCCESS: 119
- MALFORMED: 1
- Malformed instance(s): ADSO_TEST_050

### Ablations
- Rows: 360
- Status: Counter({'SUCCESS': 360})

## Qwen3-8B

### C0-C3
- Rows: 480
- Status: Counter({'SUCCESS': 480})

### E1
- Rows: 120
- SUCCESS: 118
- MALFORMED: 2
- Malformed instance(s): ADSO_TEST_055, ADSO_TEST_083

### Ablations
- Rows: 360
- Status: Counter({'SUCCESS': 360})

## Important reproducibility rule

The public CSV files do not silently convert malformed outputs into successful model decisions. Status is retained explicitly. Raw canonical JSONL is included under `raw/` for auditing.

Any downstream metric script must document how malformed rows are handled.
