# ADSO Llama-3.1-8B-Instruct Held-out Ablation Runner v1
# -------------------------------------------------------
# Uses the SAME frozen Llama E1 from the original held-out experiment.
# Does NOT regenerate E1.
#
# New component-removal conditions:
#   A_NO_E1_E2       = E0 + E2
#   A_NO_E1_E2_E3    = E0 + E2 + E3
#   B_NO_E2_E1_E3    = E0 + frozen Llama E1 + E3
#
# Total new generations: 120 x 3 = 360
#
# Run INSIDE the same Colab namespace where `model` and `tokenizer`
# are already loaded:
#
# exec(open("/content/ADSO_Llama31_Ablation_Runner_v1.py").read(), globals())
#
# Required uploaded files:
#   ADSO_Llama31_C0_C3_PublicManifest_v1.json
#   ADSO_Final_Llama31_8B_E1_Results.jsonl
#   ADSO_Llama31_Ablation_Runner_v1.py
#
# Scientific rules:
# - PRIVATE Ground Truth is never loaded.
# - Fresh one-message context per generation.
# - Existing frozen Llama E1 is reused verbatim.
# - MALFORMED E1 is preserved verbatim; no E1 repair/rerun.
# - Decision MALFORMED outputs are preserved; no selective rerun.
# - Checkpoint after every generated decision.
# - do_sample=False, same deterministic generation style as official rerun.

from pathlib import Path
import json, os, time, datetime, traceback
import torch

EXPECTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MANIFEST_NAME = "ADSO_Llama31_C0_C3_PublicManifest_v1.json"
E1_NAME = "ADSO_Final_Llama31_8B_E1_Results.jsonl"

DRIVE_OUT = Path("/content/drive/MyDrive/ADSO_Llama31_Ablation")
LOCAL_OUT = Path("/content/ADSO_Llama31_Ablation")

VARIANTS = ("A_NO_E1_E2", "A_NO_E1_E2_E3", "B_NO_E2_E1_E3")
MAX_NEW_TOKENS = 256
SEED = 20260916

if "model" not in globals() or "tokenizer" not in globals():
    raise RuntimeError(
        "Load Llama model/tokenizer first and run this file with "
        "exec(..., globals()), not !python."
    )

loaded_model_id = str(getattr(getattr(model, "config", None), "_name_or_path", "UNKNOWN"))
print("Loaded model:", loaded_model_id)

if "llama" not in loaded_model_id.lower() or "3.1" not in loaded_model_id.lower():
    raise RuntimeError(
        f"Loaded model does not look like Meta-Llama-3.1-8B-Instruct: {loaded_model_id}"
    )

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    model.eval()
except Exception:
    pass

OUT_ROOT = DRIVE_OUT if Path("/content/drive/MyDrive").exists() else LOCAL_OUT
OUT_ROOT.mkdir(parents=True, exist_ok=True)

if OUT_ROOT == LOCAL_OUT:
    print("WARNING: Google Drive is not mounted. Output is only in /content.")
    print("Mount Drive before continuing if you want disconnect-safe checkpoints.")

def locate(name):
    for p in (Path("/content") / name, Path.cwd() / name, OUT_ROOT / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find required file: {name}")

manifest_path = locate(MANIFEST_NAME)
e1_path = locate(E1_NAME)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("contains_private_ground_truth") is not False:
    raise RuntimeError("Manifest leakage guard failed.")

instances = manifest["instances"]
if len(instances) != 120:
    raise RuntimeError(f"Expected 120 instances; found {len(instances)}")

def read_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL {path}:{n}: {e}")
    return out

def load_e1(path):
    rows = read_jsonl(path)
    d = {}
    for r in rows:
        iid = r.get("instance_id")
        if not iid or not str(iid).startswith("ADSO_TEST_"):
            continue
        if iid in d:
            raise RuntimeError(f"Duplicate E1 for {iid}")

        raw = r.get("raw_response")
        if raw is None:
            raise RuntimeError(f"Missing raw_response for {iid}")

        # Preserve original E1 verbatim, including the one originally MALFORMED.
        d[iid] = {
            "raw_response": str(raw),
            "status": r.get("status"),
            "parse_method": r.get("parse_method"),
        }

    expected = {f"ADSO_TEST_{i:03d}" for i in range(1, 121)}
    if set(d) != expected:
        raise RuntimeError(
            "E1 must contain exactly all 120 held-out Llama instances. "
            f"Missing={sorted(expected-set(d))[:10]}"
        )
    return d

e1 = load_e1(e1_path)
status_counts = {}
for v in e1.values():
    status_counts[v["status"]] = status_counts.get(v["status"], 0) + 1

print("Frozen Llama E1 loaded:", len(e1), "/120")
print("E1 status counts:", status_counts)
print("No E1 regeneration will occur.")

RESULTS = OUT_ROOT / "ADSO_Llama31_Ablation_360_Results.jsonl"
META = OUT_ROOT / "ADSO_Llama31_Ablation_Metadata.json"

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def valid_decision(obj):
    return (
        isinstance(obj, dict)
        and set(obj.keys()) == {"decision", "confidence", "reason"}
        and obj.get("decision") in {"OPTIMIZE", "PRESERVE"}
        and isinstance(obj.get("confidence"), int)
        and not isinstance(obj.get("confidence"), bool)
        and 0 <= obj["confidence"] <= 100
        and isinstance(obj.get("reason"), str)
    )

def balanced_json_objects(text):
    depth = 0
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
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
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i+1]
                start = None

def parse_first_valid_json(raw):
    s = raw.strip()
    try:
        obj = json.loads(s)
        if valid_decision(obj):
            return obj, "direct_json"
    except Exception:
        pass

    for candidate in balanced_json_objects(raw):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if valid_decision(obj):
            return obj, "first_valid_json_extracted"

    return None, "no_valid_json"

def input_device():
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEVICE = input_device()

if getattr(tokenizer, "pad_token_id", None) is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    enc = tokenizer(rendered, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    input_tokens = int(enc["input_ids"].shape[1])

    started = time.perf_counter()

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - started
    generated = out[0, enc["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    output_tokens = int(generated.shape[0])

    del enc, out, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw, input_tokens, output_tokens, elapsed

def build_prompt(rec, variant):
    parts = [
        rec["decision_prefix"].rstrip(),
        "Additional diagnostic evidence is supplied below. Use it as evidence, "
        "but make the final OPTIMIZE/PRESERVE decision yourself."
    ]

    iid = rec["instance_id"]

    if variant == "A_NO_E1_E2":
        parts.append(
            "E2 — deterministic neutral static-analysis evidence:\n"
            + rec["e2_text"]
        )

    elif variant == "A_NO_E1_E2_E3":
        parts.append(
            "E2 — deterministic neutral static-analysis evidence:\n"
            + rec["e2_text"]
        )
        parts.append(
            "E3 — neutral per-program dynamic execution evidence:\n"
            + rec["e3_text"]
        )

    elif variant == "B_NO_E2_E1_E3":
        parts.append(
            "E1 — this model's own previously generated structured self-diagnosis:\n"
            + e1[iid]["raw_response"]
        )
        parts.append(
            "E3 — neutral per-program dynamic execution evidence:\n"
            + rec["e3_text"]
        )

    else:
        raise ValueError(variant)

    parts.append(rec["decision_suffix"])
    return "\n\n".join(parts)

existing = read_jsonl(RESULTS)
done = {(r["instance_id"], r["variant"]) for r in existing}

print("Checkpoint:", len(done), "/360 already complete.")

metadata = {
    "runner_version": "1.0",
    "model": EXPECTED_MODEL,
    "loaded_model": loaded_model_id,
    "e1_filename": e1_path.name,
    "e1_reused_frozen": True,
    "e1_regenerated": False,
    "contains_private_ground_truth": False,
    "variants": list(VARIANTS),
    "expected_calls": 360,
    "seed": SEED,
    "do_sample": False,
    "max_new_tokens": MAX_NEW_TOKENS,
    "parser": "deterministic first valid JSON extraction",
    "output_root": str(OUT_ROOT),
    "started_utc": now_utc(),
}
META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

counter = 0

for rec in instances:
    iid = rec["instance_id"]

    for variant in VARIANTS:
        counter += 1
        if (iid, variant) in done:
            continue

        print(f"[{counter:03d}/360] {iid} {variant} running...")

        try:
            prompt = build_prompt(rec, variant)
            raw, in_tok, out_tok, secs = generate(prompt)
            parsed, parse_method = parse_first_valid_json(raw)
            status = "SUCCESS" if parsed else "MALFORMED"

            row = {
                "instance_id": iid,
                "variant": variant,
                "model": EXPECTED_MODEL,
                "stage": "ABLATION_FINAL_DECISION",
                "status": status,
                "decision": parsed["decision"] if parsed else None,
                "confidence": parsed["confidence"] if parsed else None,
                "reason": parsed["reason"] if parsed else None,
                "raw_response": raw,
                "parse_method": parse_method,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_seconds": round(secs, 4),
                "timestamp_utc": now_utc(),
            }

            append_jsonl(RESULTS, row)
            done.add((iid, variant))

            if parsed:
                print(
                    f"    SUCCESS {parsed['decision']} "
                    f"confidence={parsed['confidence']} ({parse_method})"
                )
            else:
                print(f"    MALFORMED preserved verbatim ({parse_method})")

        except KeyboardInterrupt:
            print("\nInterrupted. Drive checkpoint is safe. Re-run to resume.")
            raise
        except Exception:
            print("\nRuntime failure. Completed rows remain checkpointed.")
            print("Reconnect/reload model and re-run this exact runner to resume.")
            traceback.print_exc()
            raise

print("\nCOMPLETE: 360/360")
print("Results:", RESULTS)
print("Metadata:", META)
print("PRIVATE Ground Truth was never loaded.")
