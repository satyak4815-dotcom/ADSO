# ADSO GPT-OSS-120B Held-out Ablation Runner v1.1
# ------------------------------------------------
# Runs only the genuinely new component-removal conditions:
#   A_NO_E1_E2      = E0 + E2
#   A_NO_E1_E2_E3   = E0 + E2 + E3
#   B_NO_E2_E1_E3   = E0 + frozen GPT-OSS E1 + E3
#
# Total new calls: 120 * 3 = 360.
#
# IMPORTANT SCIENTIFIC RULES
# - Uses openai/gpt-oss-120b via Groq.
# - PRIVATE Ground Truth is never loaded.
# - Fresh independent request for every instance/condition.
# - The same GPT-OSS E1 used in the original C1-C3 held-out run MUST be supplied.
# - Do NOT regenerate E1 for this ablation.
# - Malformed scientific outputs are recorded as MALFORMED and are NOT retried.
# - Retry only technical/API/network failures.
# - Results are checkpointed after every completed generation.
#
# Required uploaded files:
#   ADSO_GPTOSS_Ablation_PublicManifest_v1.json
#   GPT-OSS held-out E1 JSONL from the original experiment (120 instances)
#
# Required environment:
#   GROQ_API_KEY
#
# Install:
#   !pip -q install -U groq
#
# Run:
#   %run /content/ADSO_GPTOSS_Ablation_Runner_v1.py \
#       --manifest /content/ADSO_GPTOSS_Ablation_PublicManifest_v1.json \
#       --e1 /content/YOUR_GPTOSS_E1_RESULTS.jsonl \
#       --outdir /content/drive/MyDrive/ADSO_GPTOSS_Ablation

import argparse, json, os, re, time, random, hashlib
from pathlib import Path
from datetime import datetime, timezone

MODEL = "openai/gpt-oss-120b"
PROVIDER = "Groq"
VARIANTS = ("A_NO_E1_E2", "A_NO_E1_E2_E3", "B_NO_E2_E1_E3")
MAX_TECHNICAL_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 5.0

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL {path}:{ln}: {e}")
    return out

def first_valid_decision(raw):
    # 1) direct JSON
    try:
        obj = json.loads(raw.strip())
        if validate_decision(obj):
            return obj, "direct_json"
    except Exception:
        pass

    # 2) first balanced valid JSON object
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if validate_decision(obj):
                        return obj, "first_valid_json_extracted"
                except Exception:
                    pass
                start = None
    return None, "no_valid_json"

def validate_decision(obj):
    return (
        isinstance(obj, dict)
        and set(obj.keys()) == {"decision", "confidence", "reason"}
        and obj.get("decision") in {"OPTIMIZE", "PRESERVE"}
        and isinstance(obj.get("confidence"), int)
        and not isinstance(obj.get("confidence"), bool)
        and 0 <= obj["confidence"] <= 100
        and isinstance(obj.get("reason"), str)
    )

def load_e1(path):
    """
    Recover exactly one successful original GPT-OSS E1 response per ADSO_TEST instance.

    The original audit JSONL may contain technical API_ERROR rows followed later by a
    successful retry for the same instance. Those technical-failure rows are part of
    the audit trail but are not scientific E1 responses, so they are skipped here.
    No successful E1 response is regenerated or altered.
    """
    rows = load_jsonl(path)
    d = {}
    success_rows_seen = {}

    for n, x in enumerate(rows, 1):
        iid = x.get("instance_id")
        if not iid or not str(iid).startswith("ADSO_TEST_"):
            continue

        status = str(x.get("status", "")).upper()

        # Technical failures are audit records, not model E1 evidence.
        if status in {"API_ERROR", "ERROR", "FAILED", "TECHNICAL_ERROR"}:
            continue

        # Accept the original successful E1 formats while preserving raw content.
        raw = x.get("raw_response")
        if raw is None:
            raw = x.get("raw")

        if raw is None:
            # Some audit formats store the structured E1 fields directly in the row.
            e1_keys = [
                "time_complexity",
                "auxiliary_space_complexity",
                "main_bottleneck",
                "avoidable_work",
                "constraint_fit",
                "optimality_rationale",
                "diagnostic_confidence",
            ]
            if all(k in x for k in e1_keys):
                raw = json.dumps({k: x[k] for k in e1_keys}, ensure_ascii=False)

        if raw is None:
            parsed = x.get("parsed")
            if parsed is not None:
                raw = json.dumps(parsed, ensure_ascii=False)

        if raw is None or not str(raw).strip():
            # A non-technical row without a usable response is not silently repaired.
            continue

        if iid in d:
            raise RuntimeError(
                f"Multiple usable successful GPT-OSS E1 responses found for {iid}. "
                "Refusing to choose selectively."
            )

        d[iid] = str(raw)
        success_rows_seen[iid] = n

    expected = {f"ADSO_TEST_{i:03d}" for i in range(1, 121)}
    if set(d) != expected:
        missing = sorted(expected - set(d))
        extra = sorted(set(d) - expected)
        raise RuntimeError(
            f"E1 audit log must yield exactly one usable successful original E1 response "
            f"for each of 120 held-out instances. Missing={missing[:10]}, extra={extra[:10]}"
        )

    print(f"Loaded original successful GPT-OSS E1 responses: {len(d)}/120")
    return d

def build_prompt(rec, variant, e1_raw):
    parts = [rec["base_prompt"]]
    parts.append(
        "Additional diagnostic evidence is supplied below. Use it as evidence, "
        "but make the final OPTIMIZE/PRESERVE decision yourself."
    )

    if variant == "A_NO_E1_E2":
        parts.append("E2 — deterministic neutral static-analysis evidence:\n" + rec["e2_text"])
    elif variant == "A_NO_E1_E2_E3":
        parts.append("E2 — deterministic neutral static-analysis evidence:\n" + rec["e2_text"])
        parts.append("E3 — neutral per-program dynamic execution evidence:\n" + rec["e3_text"])
    elif variant == "B_NO_E2_E1_E3":
        parts.append("E1 — this model's own previously generated structured self-diagnosis:\n" + e1_raw)
        parts.append("E3 — neutral per-program dynamic execution evidence:\n" + rec["e3_text"])
    else:
        raise ValueError(variant)

    parts.append(rec["decision_suffix"])
    return "\n\n".join(parts)

def usage_value(usage, name):
    v = getattr(usage, name, None)
    return int(v) if v is not None else None

def run_api(client, prompt):
    # Provider defaults are intentionally retained for sampling parameters,
    # matching the frozen protocol unless an earlier run explicitly froze otherwise.
    last_error = None
    for attempt in range(1, MAX_TECHNICAL_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.perf_counter() - started
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            return {
                "raw": raw,
                "attempt": attempt,
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": usage_value(usage, "prompt_tokens") if usage else None,
                "completion_tokens": usage_value(usage, "completion_tokens") if usage else None,
                "total_tokens": usage_value(usage, "total_tokens") if usage else None,
            }
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = repr(e)
            if attempt >= MAX_TECHNICAL_ATTEMPTS:
                break
            delay = min(60.0, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(f"      technical failure attempt {attempt}: {e}")
            print(f"      retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError(f"Technical API failure after {MAX_TECHNICAL_ATTEMPTS} attempts: {last_error}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from groq import Groq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY before running.")

    manifest_path = Path(args.manifest)
    e1_path = Path(args.e1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("contains_private_ground_truth") is not False:
        raise RuntimeError("Manifest leakage guard failed.")

    instances = manifest["instances"]
    if len(instances) != 120:
        raise RuntimeError(f"Expected 120 manifest instances, got {len(instances)}")

    e1 = load_e1(e1_path)

    out_file = outdir / "ADSO_GPTOSS_Ablation_360_Results.jsonl"
    meta_file = outdir / "ADSO_GPTOSS_Ablation_Metadata.json"

    existing = load_jsonl(out_file)
    done = {(x.get("instance_id"), x.get("variant")) for x in existing}
    print(f"Checkpoint: {len(done)}/360 already complete.")

    metadata = {
        "runner_version": "1.0",
        "model": MODEL,
        "provider": PROVIDER,
        "manifest_sha256": manifest_sha,
        "e1_filename": e1_path.name,
        "contains_private_ground_truth": False,
        "variants": list(VARIANTS),
        "expected_calls": 360,
        "sampling": "Groq/provider defaults; no temperature/top_p override",
        "technical_retry_limit": MAX_TECHNICAL_ATTEMPTS,
        "malformed_retry": False,
        "started_utc": now_utc(),
    }
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    client = Groq()
    counter = 0

    for rec in instances:
        iid = rec["instance_id"]
        for variant in VARIANTS:
            counter += 1
            key = (iid, variant)
            if key in done:
                continue

            prompt = build_prompt(rec, variant, e1[iid])
            print(f"[{counter:03d}/360] {iid} {variant} running...")

            try:
                api = run_api(client, prompt)
            except KeyboardInterrupt:
                print("\nInterrupted. Existing checkpoint is safe; rerun to resume.")
                raise
            except Exception:
                print("\nTechnical failure. No scientific response written for this run.")
                print("Fix API/connectivity/quota and rerun; checkpoint resume is automatic.")
                raise

            parsed, parse_mode = first_valid_decision(api["raw"])
            status = "SUCCESS" if parsed else "MALFORMED"

            row = {
                "instance_id": iid,
                "variant": variant,
                "model": MODEL,
                "provider": PROVIDER,
                "stage": "ABLATION_FINAL_DECISION",
                "status": status,
                "decision": parsed["decision"] if parsed else None,
                "confidence": parsed["confidence"] if parsed else None,
                "reason": parsed["reason"] if parsed else None,
                "raw_response": api["raw"],
                "parse_method": parse_mode,
                "attempt": api["attempt"],
                "latency_seconds": api["latency_seconds"],
                "prompt_tokens": api["prompt_tokens"],
                "completion_tokens": api["completion_tokens"],
                "total_tokens": api["total_tokens"],
                "timestamp_utc": now_utc(),
            }
            append_jsonl(out_file, row)
            done.add(key)

            if parsed:
                print(f"      SUCCESS {parsed['decision']} confidence={parsed['confidence']} ({parse_mode})")
            else:
                print(f"      MALFORMED preserved verbatim ({parse_mode})")

    print("\nAblation collection complete: 360/360.")
    print("Results:", out_file)
    print("Metadata:", meta_file)
    print("PRIVATE Ground Truth was never loaded.")

if __name__ == "__main__":
    main()
