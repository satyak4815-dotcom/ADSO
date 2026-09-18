# ADSO Qwen3-8B Held-out Main Experiment Runner v1
# 120 E1 generations + 480 C0-C3 decisions = 600 scientific generations.
#
# IMPORTANT:
# - Run inside the SAME Colab namespace where `model` and `tokenizer` are loaded.
# - Qwen3 thinking mode is DISABLED consistently.
# - No private ground truth is loaded.
# - E1 is generated once per instance, checkpointed, then reused verbatim.
# - Fresh one-user-message context for every generation.
# - Scientific malformed outputs are preserved, not selectively rerun.
# - Drive checkpoint after every generation.
#
# Run:
# exec(open("/content/ADSO_Qwen3_8B_Heldout_Runner_v1.py").read(), globals())

from pathlib import Path
import json, os, time, datetime, traceback, hashlib
import torch

EXPECTED_MODEL = "Qwen/Qwen3-8B"
MANIFEST_NAME = "ADSO_Qwen3_8B_Heldout_PublicManifest_v1.json"
DRIVE_OUT = Path("/content/drive/MyDrive/ADSO_Qwen3_8B_Heldout")
LOCAL_OUT = Path("/content/ADSO_Qwen3_8B_Heldout")

E1_FILE = "ADSO_Final_Qwen3_8B_E1_Results.jsonl"
DEC_FILE = "ADSO_Final_Qwen3_8B_C0_C3_Results.jsonl"
META_FILE = "ADSO_Final_Qwen3_8B_Metadata.json"

SEED = 20260916
MAX_NEW_E1 = 512
MAX_NEW_DECISION = 256

if "model" not in globals() or "tokenizer" not in globals():
    raise RuntimeError("Load Qwen3-8B model/tokenizer first, then use exec(..., globals()).")

loaded_model = str(getattr(getattr(model, "config", None), "_name_or_path", "UNKNOWN"))
print("Loaded model:", loaded_model)
if "qwen3" not in loaded_model.lower() or "8b" not in loaded_model.lower():
    raise RuntimeError(f"Unexpected loaded model: {loaded_model}")

if not Path("/content/drive/MyDrive").exists():
    raise RuntimeError("Google Drive is not mounted. Mount Drive before the scientific run.")

DRIVE_OUT.mkdir(parents=True, exist_ok=True)
OUT = DRIVE_OUT

manifest_path = Path("/content") / MANIFEST_NAME
if not manifest_path.exists():
    raise FileNotFoundError(f"Upload {MANIFEST_NAME} to /content first.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("contains_private_ground_truth") is not False:
    raise RuntimeError("Leakage guard failed: manifest must explicitly exclude private ground truth.")
instances = manifest["instances"]
if len(instances) != 120:
    raise RuntimeError(f"Expected 120 held-out instances, got {len(instances)}.")

E1_PATH = OUT / E1_FILE
DEC_PATH = OUT / DEC_FILE
META_PATH = OUT / META_FILE

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise RuntimeError(f"Corrupt checkpoint {path}:{n}: {e}")
    return rows

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def balanced_json_objects(text):
    depth, start, in_string, escaped = 0, None, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped: escaped = False
            elif ch == "\\": escaped = True
            elif ch == '"': in_string = False
            continue
        if ch == '"': in_string = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i+1]
                start = None

E1_KEYS = {
    "time_complexity", "auxiliary_space_complexity", "main_bottleneck",
    "avoidable_work", "constraint_fit", "optimality_rationale",
    "diagnostic_confidence"
}

def valid_e1(o):
    return (
        isinstance(o, dict)
        and set(o.keys()) == E1_KEYS
        and isinstance(o["diagnostic_confidence"], int)
        and not isinstance(o["diagnostic_confidence"], bool)
        and 0 <= o["diagnostic_confidence"] <= 100
        and all(isinstance(o[k], str) for k in E1_KEYS - {"diagnostic_confidence"})
    )

def valid_decision(o):
    return (
        isinstance(o, dict)
        and set(o.keys()) == {"decision", "confidence", "reason"}
        and o["decision"] in {"OPTIMIZE", "PRESERVE"}
        and isinstance(o["confidence"], int)
        and not isinstance(o["confidence"], bool)
        and 0 <= o["confidence"] <= 100
        and isinstance(o["reason"], str)
    )

def parse_json(raw, validator):
    s = raw.strip()
    try:
        o = json.loads(s)
        if validator(o): return o, "direct_json"
    except Exception:
        pass
    for c in balanced_json_objects(raw):
        try: o = json.loads(c)
        except Exception: continue
        if validator(o): return o, "first_valid_json_extracted"
    return None, "no_valid_json"

try:
    DEVICE = next(model.parameters()).device
except Exception:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate(prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    # Qwen3: freeze NON-THINKING mode for every scientific generation.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tokenizer(rendered, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    sec = time.perf_counter() - t0
    gen = out[0, n_in:]
    raw = tokenizer.decode(gen, skip_special_tokens=True)
    n_out = int(gen.shape[0])
    del enc, out, gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw, n_in, n_out, sec

metadata = {
    "runner_version": "1.0",
    "model": EXPECTED_MODEL,
    "loaded_model": loaded_model,
    "manifest_filename": MANIFEST_NAME,
    "manifest_sha256": sha256(manifest_path),
    "contains_private_ground_truth": False,
    "thinking_mode": "disabled",
    "thinking_control": "tokenizer.apply_chat_template(enable_thinking=False)",
    "sampling": {"do_sample": False, "seed": SEED},
    "max_new_tokens": {"E1": MAX_NEW_E1, "decision": MAX_NEW_DECISION},
    "fresh_context_each_generation": True,
    "e1_generated_once_and_reused": True,
    "scientific_malformed_retry": False,
    "expected_generations": {"E1": 120, "C0_C3": 480, "total": 600},
    "started_utc": utcnow(),
}
META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# ---------------- PHASE 1: E1 ----------------
e1_rows = read_jsonl(E1_PATH)
e1_done = {r["instance_id"]: r for r in e1_rows}
print(f"E1 checkpoint: {len(e1_done)}/120 complete")

for idx, rec in enumerate(instances, 1):
    iid = rec["instance_id"]
    if iid in e1_done:
        continue
    print(f"[E1 {idx:03d}/120] {iid} running...")
    try:
        raw, ni, no, sec = generate(rec["e1_prompt"], MAX_NEW_E1)
        parsed, method = parse_json(raw, valid_e1)
        row = {
            "instance_id": iid,
            "model": EXPECTED_MODEL,
            "stage": "E1",
            "status": "SUCCESS" if parsed else "MALFORMED",
            "parsed": parsed,
            "raw_response": raw,
            "parse_method": method,
            "input_tokens": ni,
            "output_tokens": no,
            "latency_seconds": round(sec, 4),
            "timestamp_utc": utcnow(),
        }
        append_jsonl(E1_PATH, row)
        e1_done[iid] = row
        print("   ", row["status"], method)
    except KeyboardInterrupt:
        print("\nInterrupted; E1 checkpoint is safe.")
        raise
    except Exception:
        print("\nRuntime/technical failure; completed E1 rows are safe. Re-run to resume.")
        traceback.print_exc()
        raise

if len(e1_done) != 120:
    raise RuntimeError("E1 phase incomplete.")

print("\nE1 COMPLETE: 120/120. Frozen for C1-C3.")

def build_decision_prompt(rec, cond):
    iid = rec["instance_id"]
    parts = [rec["decision_prefix"].rstrip()]
    if cond in {"C1", "C2", "C3"}:
        # Reuse exact original Qwen E1 response verbatim, including MALFORMED output.
        parts.append(
            "E1 — this model's own previously generated structured self-diagnosis:\n"
            + e1_done[iid]["raw_response"]
        )
    if cond in {"C2", "C3"}:
        parts.append(
            "E2 — deterministic neutral static-analysis evidence:\n" + rec["e2_text"]
        )
    if cond == "C3":
        parts.append(
            "E3 — neutral per-program dynamic execution evidence:\n" + rec["e3_text"]
        )
    parts.append(rec["decision_suffix"])
    return "\n\n".join(parts)

# ---------------- PHASE 2: C0-C3 ----------------
dec_rows = read_jsonl(DEC_PATH)
done = {(r["instance_id"], r["condition"]) for r in dec_rows}
print(f"C0-C3 checkpoint: {len(done)}/480 complete")

counter = 0
for rec in instances:
    iid = rec["instance_id"]
    for cond in ("C0", "C1", "C2", "C3"):
        counter += 1
        if (iid, cond) in done:
            continue
        print(f"[DEC {counter:03d}/480] {iid} {cond} running...")
        try:
            prompt = build_decision_prompt(rec, cond)
            raw, ni, no, sec = generate(prompt, MAX_NEW_DECISION)
            parsed, method = parse_json(raw, valid_decision)
            row = {
                "instance_id": iid,
                "condition": cond,
                "model": EXPECTED_MODEL,
                "stage": "FINAL_DECISION",
                "status": "SUCCESS" if parsed else "MALFORMED",
                "decision": parsed["decision"] if parsed else None,
                "confidence": parsed["confidence"] if parsed else None,
                "reason": parsed["reason"] if parsed else None,
                "raw_response": raw,
                "parse_method": method,
                "input_tokens": ni,
                "output_tokens": no,
                "latency_seconds": round(sec, 4),
                "timestamp_utc": utcnow(),
            }
            append_jsonl(DEC_PATH, row)
            done.add((iid, cond))
            if parsed:
                print(f"    SUCCESS {parsed['decision']} confidence={parsed['confidence']} ({method})")
            else:
                print(f"    MALFORMED preserved ({method})")
        except KeyboardInterrupt:
            print("\nInterrupted; C0-C3 checkpoint is safe.")
            raise
        except Exception:
            print("\nRuntime/technical failure; completed decision rows are safe. Re-run to resume.")
            traceback.print_exc()
            raise

print("\nMAIN QWEN EXPERIMENT COMPLETE")
print("E1:", E1_PATH)
print("C0-C3:", DEC_PATH)
print("Metadata:", META_PATH)
print("Expected total scientific generations: 600")
print("Private ground truth was never loaded.")
