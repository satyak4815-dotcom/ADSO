# Frozen Evaluation Guard

| Rule | Requirement |
|---|---|
| Fresh context | Each instance in a fresh model context/chat. |
| E1 ownership | Each model generates its own E1; never reuse another model's E1. |
| E2/E3 | Neutral E2/E3 evidence may be shared across models. |
| No content retries | Retry only transport/interface failures, never because an answer looks wrong. |
| Raw output | Preserve exact responses and malformed outputs. |
| No leakage | Never expose PRIVATE Ground Truth, pair role, paired code, or relative pair speedup. |
| Policy freeze | Use ADSO Policy v1.5 unchanged. |
| No test tuning | Do not alter prompts, thresholds, labels, or evidence definitions after viewing final-test model performance. |
