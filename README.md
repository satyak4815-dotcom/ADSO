# ADSO: Adaptive Diagnostic Sufficiency for Optimization

This repository contains the frozen held-out benchmark and reproducibility artifacts for the paper:

**Adaptive Diagnostic Sufficiency for Reliable LLM-Based Code Optimization**

Authors: Satya Krishna Singh and Archisha Singh  
Affiliation: Department of Computer Science and Engineering, Noida Institute of Engineering and Technology, Greater Noida, India

## What this repository contains

ADSO studies a decision that comes *before* code optimization: whether a functionally correct implementation should be **OPTIMIZED** or **PRESERVED**, and how much diagnostic evidence an LLM requires before making that decision.

The final held-out benchmark contains:

- 60 unique programming problems
- 120 program instances
- 60 OPTIMIZE labels
- 60 PRESERVE labels
- 41 Level-A pairs
- 19 Level-B pairs
- 640/640 successful program-test executions
- 0 runtime errors/timeouts in the correctness audit
- neutral randomized IDs `ADSO_TEST_001` ... `ADSO_TEST_120`
- frozen randomization seed: `20260915`

## Evidence hierarchy

- **E0** — source code, problem specification, and constraints
- **E1** — structured self-diagnosis generated separately by each evaluated model
- **E2** — deterministic static-analysis evidence
- **E3** — controlled absolute dynamic profiling evidence

The cumulative fixed conditions are:

- `C0 = E0`
- `C1 = E0 + E1`
- `C2 = E0 + E1 + E2`
- `C3 = E0 + E1 + E2 + E3`

## Repository layout

```text
dataset/
  benchmark_summary.csv
  pair_validation.csv
  benchmark_instances.csv
  e0_prompts.csv
  e2_static_evidence.csv
  e3_dynamic_evidence.csv
  ground_truth.csv

protocol/
  adso_policy.md
  evidence_definitions.md
  evaluation_guard.md

provenance/
  README.md

results/
  README.md

scripts/
figures/
MANIFEST.csv
CITATION.cff
RELEASE_CHECKLIST.md
```

## Important leakage rule

`dataset/ground_truth.csv` is released **only for post-evaluation reproducibility**. It must never be supplied to an evaluated model. During the reported experiments, models did not receive the private label, hidden pair role, counterpart implementation, or pair-relative speedup.

## Dataset provenance

The held-out benchmark is **derived from the ACEOB test split** and is not presented as an independently created replacement for ACEOB.

Upstream ACEOB repository:
https://github.com/CodeGeneration2/ACEOB

ACEOB paper:
Y. Pan, X. Shao, and C. Lyu, "Measuring code efficiency optimization capabilities with ACEOB," Journal of Systems and Software, vol. 219, 112250, 2025.
https://doi.org/10.1016/j.jss.2024.112250

The upstream ACEOB GitHub repository identifies an MIT license. Original Codeforces problem URLs are retained in the released tables for provenance.

## Development/test separation

An earlier 8-pair / 16-instance benchmark was used only during development and policy freezing. It is not reused as part of the final 120-instance held-out evaluation.

## Reproducibility status

This package contains the final benchmark-side artifacts. Cleaned model outputs and ablation results should be added under `results/` only after they are matched against the canonical 120 frozen instance IDs and retry/duplicate rows are separated from final observations.


## Model outputs included in this release

The repository also includes cleaned held-out outputs for:

- GPT-OSS-120B
- Meta-Llama-3.1-8B-Instruct
- Qwen3-8B

For each model, the public package includes:
- C0-C3 fixed-condition outputs,
- model-specific E1 self-diagnosis outputs,
- three controlled ablation variants,
- configuration metadata,
- and raw canonical JSONL for audit.

Malformed outputs and technical retries are documented transparently in `results/DATA_QUALITY.md`.
