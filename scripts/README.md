# Reproduction Scripts

These scripts are copied from the final experiment bundle and are included to document the actual model-running workflow.

## Included

- `adso_final_gptoss_c0_c3_runner_RATE_LIMIT_SAFE.py`
- `adso_final_gptoss_e1_runner_RATE_LIMIT_SAFE.py`
- `ADSO_GPTOSS_Ablation_Runner_v1.1.py`
- `ADSO_Final_Llama31_8B_C0_C3_Colab_Runner_v2.py`
- `adso_final_llama31_8b_e1_colab_runner.py`
- `ADSO_Llama31_Ablation_Runner_v1.py`
- `ADSO_Qwen3_8B_Heldout_Runner_v1.py`
- `ADSO_Qwen3_8B_Ablation_Runner_v1.py`

## Secret handling

The public repository must never contain API keys or access tokens.

Automated obvious-secret scan result for the copied scripts: **PASS — no obvious hard-coded secret detected**.

GPT-OSS runners are expected to obtain the Groq credential from an environment variable rather than embedding it in source code.

## Environment note

Some scripts were originally run in Google Colab and may contain local/Drive output paths. These paths are execution-environment details, not credentials. Users reproducing the experiment should update local file paths to match their environment.
