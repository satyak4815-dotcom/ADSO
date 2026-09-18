# Release Checklist

## Benchmark package
- [x] Frozen 60-problem / 120-instance held-out benchmark
- [x] 60 OPTIMIZE / 60 PRESERVE labels
- [x] E0, E2, and E3 evidence exported
- [x] Ground truth isolated from model-visible files
- [x] Frozen ADSO policy documented
- [x] Evaluation guard documented

## Model experiment cleanup
- [x] GPT C0-C3 restricted to 120 canonical held-out IDs
- [x] GPT E1 technical API retries separated from final E1 observations
- [x] GPT malformed ablation rows preserved and documented
- [x] Llama C0-C3 official 480-row rerun included
- [x] Llama E1 malformed row preserved and documented
- [x] Qwen C0-C3 480-row result included
- [x] Qwen E1 malformed rows preserved and documented
- [x] Qwen thinking-disabled configuration documented
- [x] Duplicate backup/output copies excluded
- [x] Minimal final-run scripts selected
- [x] Obvious hard-coded secret scan performed

## Before public GitHub publication
- [ ] Review third-party redistribution terms for Codeforces problem statements/source material
- [ ] Decide license for ADSO-authored code and documentation
- [ ] Add final paper PDF only if conference rules permit preprint/public upload
- [ ] Add final figures if desired
- [ ] Create public GitHub repository
- [ ] Upload this cleaned package
- [ ] Replace placeholder repository URL in README/paper
- [ ] Create a GitHub release/tag (recommended: `v1.0`)
