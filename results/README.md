# Model Results

The result files in this directory are cleaned views of the final held-out model experiments.

## Structure

```text
gpt_oss/
  c0_c3_results.csv
  e1_results.csv
  configuration.json
  retry_audit/
  raw/

llama/
  c0_c3_results.csv
  e1_results.csv
  configuration.json
  raw/

qwen/
  c0_c3_results.csv
  e1_results.csv
  configuration.json
  raw/

ablations/
  gpt_oss_ablations.csv
  llama_ablations.csv
  qwen_ablations.csv
  raw/
```

## Canonical result counts

For each model:
- C0-C3: 120 instances × 4 conditions = 480 rows
- E1: 120 canonical instance rows
- Ablations: 120 instances × 3 variants = 360 rows

## Ablation variants

- `A_NO_E1_E2` = E0 + E2
- `A_NO_E1_E2_E3` = E0 + E2 + E3
- `B_NO_E2_E1_E3` = E0 + frozen same-model E1 + E3

## Malformed output policy

Malformed model outputs are retained with `status=MALFORMED`.
They are not silently repaired, relabeled, or regenerated for the public release.

See `DATA_QUALITY.md` for exact counts and exceptions.
