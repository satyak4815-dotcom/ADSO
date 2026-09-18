# ADSO Qwen3-8B Held-out Ablation Runner v1
# ------------------------------------------------
# Uses the SAME frozen Qwen E1 from the original held-out experiment.
# DOES NOT regenerate E1 and DOES NOT rerun C0-C3.
#
# New component-removal conditions:
#   A_NO_E1_E2       = E0 + E2
#   A_NO_E1_E2_E3    = E0 + E2 + E3
#   B_NO_E2_E1_E3    = E0 + frozen Qwen E1 + E3
#
# Total NEW scientific generations: 120 x 3 = 360
#
# Scientific rules:
# - PRIVATE Ground Truth is never loaded.
# - Fresh one-user-message context per generation.
# - Exact frozen Qwen E1 raw_response is reused verbatim.
# - Qwen3 thinking mode remains DISABLED, matching the main experiment.
# - do_sample=False, seed=20260916.
# - Malformed scientific responses are preserved and NOT selectively rerun.
# - Checkpoint after every generated decision.
# - Resume-safe: re-running skips completed instance/variant pairs.
#
# HOW TO RUN IN COLAB
# -------------------
# 1) Load the SAME Qwen3-8B model and tokenizer used for the main experiment.
# 2) Mount Google Drive.
# 3) Upload this file to /content.
# 4) Make sure the public manifest is available either:
#      /content/ADSO_Qwen3_8B_Heldout_PublicManifest_v1.json
#    or somewhere under the known Drive experiment folders.
# 5) The original E1 file should already exist in Drive from the main run:
#      ADSO_Final_Qwen3_8B_E1_Results.jsonl
# 6) Run:
#      exec(open("/content/ADSO_Qwen3_8B_Ablation_Runner_v1.py").read(), globals())

from pathlib import Path
import json, os, time, datetime, traceback, hashlib
import torch

EXPECTED_MODEL = "Qwen/Qwen3-8B"
MANIFEST_NAME = "ADSO_Qwen3_8B_Heldout_PublicManifest_v1.json"
E1_NAME = "ADSO_Final_Qwen3_8B_E1_Results.jsonl"

SEED = 20260916
MAX_NEW_TOKENS = 256
VARIANTS = ("A_NO_E1_E2", "A_NO_E1_E2_E3", "B_NO_E2_E1_E3")
EXPECTED_INSTANCES = 120
EXPECTED_GENERATIONS = EXPECTED_INSTANCES * len(VARIANTS)

DRIVE_ROOT = Path("/content/drive/MyDrive")
MAIN_DRIVE_DIR = DRIVE_ROOT / "ADSO_Qwen3_8B_Heldout"
ABLATION_DRIVE_DIR = DRIVE_ROOT / "ADSO_Qwen3_8B_Ablation"
LOCAL_OUT_DIR = Path("/content/ADSO_Qwen3_8B_Ablation")

# ---------------------------------------------------------------------
# ENVIRONMENT / MODEL GUARDS
# ---------------------------------------------------------------------
if "model" not in globals() or "tokenizer" not in globals():
    raise RuntimeError(
        "Qwen model/tokenizer are not loaded in this notebook namespace. "
        "Load Qwen3-8B first, then run this file with exec(..., globals())."
    )

loaded_model = str(
    getattr(getattr(model, "config", None), "_name_or_path", "UNKNOWN")
)
print("Loaded model:", loaded_model)

if "qwen3" not in loaded_model.lower() or "8b" not in loaded_model.lower():
    raise RuntimeError(
        f"Unexpected loaded model: {loaded_model}. "
        "This runner is frozen for Qwen/Qwen3-8B."
    )

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    model.eval()
except Exception:
    pass

if getattr(tokenizer, "pad_token_id", None) is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if DRIVE_ROOT.exists():
    OUT_ROOT = ABLATION_DRIVE_DIR
else:
    OUT_ROOT = LOCAL_OUT_DIR
    print("WARNING: Google Drive is not mounted.")
    print("Outputs will be checkpointed only under /content.")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------
def locate_required(name, extra_candidates=()):
    candidates = [
        Path("/content") / name,
        Path.cwd() / name,
        MAIN_DRIVE_DIR / name,
        ABLATION_DRIVE_DIR / name,
        *[Path(p) for p in extra_candidates],
    ]
    for p in candidates:
        if p.exists():
            return p

    # Conservative recursive Drive search only for the exact filename.
    if DRIVE_ROOT.exists():
        matches = list(DRIVE_ROOT.rglob(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer the official main experiment folder when possible.
            preferred = [p for p in matches if "ADSO_Qwen3_8B_Heldout" in str(p)]
            if len(preferred) == 1:
                return preferred[0]
            raise RuntimeError(
                f"Found multiple copies of {name}. "
                f"Please keep/upload the intended file in /content. Matches: "
                + ", ".join(map(str, matches[:10]))
            )

    raise FileNotFoundError(
        f"Could not find required file: {name}\n"
        f"Upload it to /content or place it in {MAIN_DRIVE_DIR}."
    )

manifest_path = locate_required(MANIFEST_NAME)
e1_path = locate_required(E1_NAME)

print("Manifest:", manifest_path)
print("Frozen E1:", e1_path)
print("Output folder:", OUT_ROOT)

# ---------------------------------------------------------------------
# HASH / TIME / JSONL HELPERS
# ---------------------------------------------------------------------
def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Corrupt JSONL {path}:{n}: {e}")
    return rows

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# ---------------------------------------------------------------------
# LOAD PUBLIC MANIFEST — NO PRIVATE GROUND TRUTH
# ---------------------------------------------------------------------
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

if manifest.get("contains_private_ground_truth") is not False:
    raise RuntimeError(
        "Leakage guard failed: public manifest must explicitly state "
        "contains_private_ground_truth=false."
    )

instances = manifest.get("instances")
if not isinstance(instances, list) or len(instances) != EXPECTED_INSTANCES:
    raise RuntimeError(
        f"Expected exactly {EXPECTED_INSTANCES} held-out instances; "
        f"found {0 if not isinstance(instances, list) else len(instances)}."
    )

required_manifest_fields = {
    "instance_id", "decision_prefix", "decision_suffix", "e2_text", "e3_text"
}
for i, rec in enumerate(instances, 1):
    missing = required_manifest_fields - set(rec.keys())
    if missing:
        raise RuntimeError(
            f"Manifest instance #{i} is missing required fields: {sorted(missing)}"
        )

instance_ids = [rec["instance_id"] for rec in instances]
if len(set(instance_ids)) != EXPECTED_INSTANCES:
    raise RuntimeError("Manifest contains duplicate instance_id values.")

# ---------------------------------------------------------------------
# LOAD AND FREEZE ORIGINAL QWEN E1
# ---------------------------------------------------------------------
def load_frozen_e1(path):
    rows = read_jsonl(path)
    by_id = {}

    for r in rows:
        iid = r.get("instance_id")
        if iid not in set(instance_ids):
            continue

        if iid in by_id:
            raise RuntimeError(f"Duplicate E1 record found for {iid}")

        if "raw_response" not in r or r["raw_response"] is None:
            raise RuntimeError(f"Frozen E1 row for {iid} is missing raw_response")

        by_id[iid] = {
            "raw_response": str(r["raw_response"]),
            "status": r.get("status"),
            "parse_method": r.get("parse_method"),
        }

    missing = [iid for iid in instance_ids if iid not in by_id]
    if missing:
        raise RuntimeError(
            "Frozen Qwen E1 is incomplete. "
            f"Missing {len(missing)} instances, first few: {missing[:10]}"
        )

    if len(by_id) != EXPECTED_INSTANCES:
        raise RuntimeError(
            f"Expected {EXPECTED_INSTANCES} frozen E1 rows; got {len(by_id)}"
        )

    return by_id

frozen_e1 = load_frozen_e1(e1_path)

status_counts = {}
for item in frozen_e1.values():
    status = item["status"]
    status_counts[status] = status_counts.get(status, 0) + 1

print(f"Frozen Qwen E1 loaded: {len(frozen_e1)}/{EXPECTED_INSTANCES}")
print("E1 status counts:", status_counts)
print("E1 WILL NOT be regenerated.")

# ---------------------------------------------------------------------
# DECISION OUTPUT PARSER
# ---------------------------------------------------------------------
def valid_decision(o):
    return (
        isinstance(o, dict)
        and set(o.keys()) == {"decision", "confidence", "reason"}
        and o.get("decision") in {"OPTIMIZE", "PRESERVE"}
        and isinstance(o.get("confidence"), int)
        and not isinstance(o.get("confidence"), bool)
        and 0 <= o["confidence"] <= 100
        and isinstance(o.get("reason"), str)
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
                yield text[start:i + 1]
                start = None

def parse_decision(raw):
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

# ---------------------------------------------------------------------
# QWEN GENERATION — EXACT SAME NON-THINKING STYLE AS MAIN RUN
# ---------------------------------------------------------------------
try:
    DEVICE = next(model.parameters()).device
except Exception:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    enc = tokenizer(rendered, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])

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
    generated = out[0, n_in:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    n_out = int(generated.shape[0])

    del enc, out, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw, n_in, n_out, elapsed

# ---------------------------------------------------------------------
# ABLATION PROMPTS
# ---------------------------------------------------------------------
def build_prompt(rec, variant):
    parts = [
        rec["decision_prefix"].rstrip(),
        (
            "Additional diagnostic evidence is supplied below. Use it as evidence, "
            "but make the final OPTIMIZE/PRESERVE decision yourself."
        ),
    ]

    iid = rec["instance_id"]

    if variant == "A_NO_E1_E2":
        # Remove E1; retain E2.
        parts.append(
            "E2 — deterministic neutral static-analysis evidence:\n"
            + rec["e2_text"]
        )

    elif variant == "A_NO_E1_E2_E3":
        # Remove E1; retain E2 and E3.
        parts.append(
            "E2 — deterministic neutral static-analysis evidence:\n"
            + rec["e2_text"]
        )
        parts.append(
            "E3 — neutral per-program dynamic execution evidence:\n"
            + rec["e3_text"]
        )

    elif variant == "B_NO_E2_E1_E3":
        # Remove E2; retain the exact same-model E1 plus E3.
        parts.append(
            "E1 — this model's own previously generated structured self-diagnosis:\n"
            + frozen_e1[iid]["raw_response"]
        )
        parts.append(
            "E3 — neutral per-program dynamic execution evidence:\n"
            + rec["e3_text"]
        )

    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    parts.append(rec["decision_suffix"])
    return "\n\n".join(parts)

# ---------------------------------------------------------------------
# OUTPUT FILES / RESUME
# ---------------------------------------------------------------------
RESULTS_PATH = OUT_ROOT / "ADSO_Qwen3_8B_Ablation_360_Results.jsonl"
META_PATH = OUT_ROOT / "ADSO_Qwen3_8B_Ablation_Metadata.json"

existing_rows = read_jsonl(RESULTS_PATH)
done = set()

for r in existing_rows:
    iid = r.get("instance_id")
    variant = r.get("variant")
    if iid in set(instance_ids) and variant in VARIANTS:
        key = (iid, variant)
        if key in done:
            raise RuntimeError(
                f"Duplicate checkpoint record detected for {iid} / {variant}. "
                "Do not continue until the checkpoint is audited."
            )
        done.add(key)

print(f"Ablation checkpoint: {len(done)}/{EXPECTED_GENERATIONS} complete.")

metadata = {
    "runner_version": "1.0",
    "experiment": "ADSO_Qwen3_8B_Heldout_Ablation",
    "model": EXPECTED_MODEL,
    "loaded_model": loaded_model,
    "manifest_filename": manifest_path.name,
    "manifest_sha256": sha256(manifest_path),
    "e1_filename": e1_path.name,
    "e1_sha256": sha256(e1_path),
    "e1_reused_frozen": True,
    "e1_regenerated": False,
    "contains_private_ground_truth": False,
    "variants": {
        "A_NO_E1_E2": "E0 + E2",
        "A_NO_E1_E2_E3": "E0 + E2 + E3",
        "B_NO_E2_E1_E3": "E0 + frozen same-model E1 + E3",
    },
    "expected_instances": EXPECTED_INSTANCES,
    "expected_new_generations": EXPECTED_GENERATIONS,
    "thinking_mode": "disabled",
    "thinking_control": "tokenizer.apply_chat_template(enable_thinking=False)",
    "sampling": {
        "do_sample": False,
        "seed": SEED,
    },
    "max_new_tokens": MAX_NEW_TOKENS,
    "fresh_context_each_generation": True,
    "scientific_malformed_retry": False,
    "resume_safe": True,
    "output_root": str(OUT_ROOT),
    "started_or_resumed_utc": utcnow(),
}
META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------
# RUN THE 360 NEW ABLATION GENERATIONS
# ---------------------------------------------------------------------
counter = 0

for rec in instances:
    iid = rec["instance_id"]

    for variant in VARIANTS:
        counter += 1
        key = (iid, variant)

        if key in done:
            continue

        print(
            f"[{counter:03d}/{EXPECTED_GENERATIONS}] "
            f"{iid} {variant} running..."
        )

        try:
            prompt = build_prompt(rec, variant)
            raw, input_tokens, output_tokens, seconds = generate(prompt)
            parsed, parse_method = parse_decision(raw)

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
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": round(seconds, 4),
                "timestamp_utc": utcnow(),
            }

            append_jsonl(RESULTS_PATH, row)
            done.add(key)

            if parsed:
                print(
                    f"    SUCCESS {parsed['decision']} "
                    f"confidence={parsed['confidence']} ({parse_method})"
                )
            else:
                print(
                    f"    MALFORMED preserved verbatim ({parse_method})"
                )

        except KeyboardInterrupt:
            print("\nInterrupted. Completed rows are safely checkpointed.")
            print("Run the same runner again to resume.")
            raise

        except Exception:
            print("\nRuntime/technical failure.")
            print("Completed rows remain safely checkpointed.")
            print("Fix/reconnect/reload the model, then run this exact runner again.")
            traceback.print_exc()
            raise

# ---------------------------------------------------------------------
# FINAL AUDIT
# ---------------------------------------------------------------------
final_rows = read_jsonl(RESULTS_PATH)
final_keys = {
    (r.get("instance_id"), r.get("variant"))
    for r in final_rows
    if r.get("instance_id") in set(instance_ids)
    and r.get("variant") in VARIANTS
}

if len(final_keys) != EXPECTED_GENERATIONS:
    raise RuntimeError(
        f"Run ended but checkpoint contains only "
        f"{len(final_keys)}/{EXPECTED_GENERATIONS} unique scientific generations."
    )

variant_counts = {
    v: sum(1 for r in final_rows if r.get("variant") == v)
    for v in VARIANTS
}
status_counts_final = {}
for r in final_rows:
    if r.get("variant") in VARIANTS:
        s = r.get("status")
        status_counts_final[s] = status_counts_final.get(s, 0) + 1

metadata["completed_utc"] = utcnow()
metadata["completed_unique_generations"] = len(final_keys)
metadata["variant_counts"] = variant_counts
metadata["status_counts"] = status_counts_final
metadata["complete"] = True
META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\n" + "=" * 72)
print("QWEN ABLATION COMPLETE")
print("=" * 72)
print(f"Unique generations: {len(final_keys)}/{EXPECTED_GENERATIONS}")
print("Variant counts:", variant_counts)
print("Status counts:", status_counts_final)
print("Results:", RESULTS_PATH)
print("Metadata:", META_PATH)
print("Frozen E1 was reused; E1 was NOT regenerated.")
print("C0-C3 were NOT rerun.")
print("PRIVATE Ground Truth was never loaded.")
