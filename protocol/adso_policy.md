# Frozen ADSO Policy

The final held-out evaluation uses a policy frozen before viewing final-test model performance.

## E0

Always escalate from E0 to E1.

## E1 stopping rule

Stop at E1 only when:

```text
D_m,0 = D_m,1
AND
min(q_m,0, q_m,1) >= 95
```

where `D` is the intervention decision and `q` is the model-reported confidence.

If this rule is not satisfied, acquire E2.

## E2 stopping rule

Stop at E2 only when:

```text
D_m,0 = D_m,1 = D_m,2
AND
mean(q_m,0, q_m,1, q_m,2) >= 85
```

If this rule is not satisfied, acquire E3.

## E3

E3 is terminal. The C3 decision is used.

## Interpretation

The confidence thresholds are operational stopping thresholds, not calibrated probabilities of correctness.

The same frozen policy is applied to all evaluated models; thresholds are not tuned to individual held-out model performance.
