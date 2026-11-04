#!/usr/bin/env bash
# Assemble and check a weights release directory for linnet-en-35m.
#
#   build/make_weights_release.sh --fp32 <linnet-en-35m.safetensors> --fp16 <linnet-en-35m.fp16.safetensors> \
#       --tokenizer <tokenizer.model> --voices <dir with *.npy> --out <dir> [--version N] [--verify-runtime] \
#       [--text "<sentence>"] [--seed N] [--temp X] [--cfg X]
#
# Flags may also come from the environment: LINNET_FP32, LINNET_FP16, LINNET_TOKENIZER, LINNET_VOICES,
# LINNET_OUT, LINNET_WEIGHTS_VERSION (default 1).
#
# Writes <out>/linnet-en-35m-weights-v<N>/ with the canonical file names
#   linnet-en-35m.safetensors        fp32 reference weights
#   linnet-en-35m.fp16.safetensors   runtime weights
#   tokenizer.model
#   voices/<name>.npy                one file per voice found in --voices (sorted by name)
#   WEIGHTS_LICENSE.md, model-card.md
#   SHA256SUMS                       `sha256sum -c SHA256SUMS` format, relative paths
#   MANIFEST.json                    version, container format, file sizes and hashes, header metadata, voices
# and checks every file first: safetensors headers (format linnet-tts/1, dtype policy: all F32 in the fp32 file,
# every 2-D tensor F16 and the rest F32 in the fp16 file, identical tensor names and metadata in both),
# voices (float32 .npy of shape [T, 48]). Python 3 standard library only.
#
# --verify-runtime builds the `synth` example of runtime/linnet-tts and synthesizes a fixed sentence with the
# fp16 weights and the first voice; the MD5 of the WAV and the exact parameters are recorded in MANIFEST.json
# under "reference" so that an x86-64 build can compare its output (the runtime's audio is bit-identical
# across thread counts and across platforms that use the same kernels).
set -euo pipefail

FP32="${LINNET_FP32:-}"
FP16="${LINNET_FP16:-}"
TOKENIZER="${LINNET_TOKENIZER:-}"
VOICES="${LINNET_VOICES:-}"
OUT="${LINNET_OUT:-}"
VERSION="${LINNET_WEIGHTS_VERSION:-1}"
VERIFY=0
TEXT="The quick brown fox jumps over the lazy dog."
SEED=0
TEMP=0.3
CFG=2.0
THREADS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --fp32) FP32="$2"; shift 2 ;;
    --fp16) FP16="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --voices) VOICES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --verify-runtime) VERIFY=1; shift ;;
    --text) TEXT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --temp) TEMP="$2"; shift 2 ;;
    --cfg) CFG="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

for v in FP32 FP16 TOKENIZER VOICES OUT; do
  [ -n "${!v}" ] || { echo "missing --$(echo "$v" | tr 'A-Z' 'a-z') (or LINNET_$v)" >&2; exit 2; }
done
for f in "$FP32" "$FP16" "$TOKENIZER"; do [ -f "$f" ] || { echo "not a file: $f" >&2; exit 1; }; done
[ -d "$VOICES" ] || { echo "not a directory: $VOICES" >&2; exit 1; }
[[ "$VERSION" =~ ^[0-9]+$ ]] || { echo "--version must be an integer: $VERSION" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="linnet-en-35m"
DEST="$OUT/$NAME-weights-v$VERSION"

mapfile -t voice_files < <(find "$VOICES" -maxdepth 1 -name '*.npy' -type f | sort)
[ "${#voice_files[@]}" -gt 0 ] || { echo "no .npy voices in $VOICES" >&2; exit 1; }

echo "== $DEST"
rm -rf "$DEST"
mkdir -p "$DEST/voices"
cp "$FP32" "$DEST/$NAME.safetensors"
cp "$FP16" "$DEST/$NAME.fp16.safetensors"
cp "$TOKENIZER" "$DEST/tokenizer.model"
for v in "${voice_files[@]}"; do cp "$v" "$DEST/voices/$(basename "$v")"; done
cp "$ROOT/WEIGHTS_LICENSE.md" "$DEST/WEIGHTS_LICENSE.md"
cp "$ROOT/docs/model-card.md" "$DEST/model-card.md"
chmod 644 "$DEST"/*.safetensors "$DEST/tokenizer.model" "$DEST"/voices/*.npy

# Validate the files and write SHA256SUMS + MANIFEST.json.
python3 - "$DEST" "$NAME" "$VERSION" <<'PY'
import ast, hashlib, json, os, struct, sys
from datetime import date

dest, name, version = sys.argv[1], sys.argv[2], int(sys.argv[3])
FORMAT = "linnet-tts/1"
VOICE_DIM = 48


def fail(msg):
    print("error:", msg, file=sys.stderr)
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 << 20:
            fail(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
    meta = header.pop("__metadata__", None)
    if not isinstance(meta, dict):
        fail(f"{path}: no __metadata__ in the header")
    if meta.get("format") != FORMAT:
        fail(f"{path}: __metadata__.format is {meta.get('format')!r}, expected {FORMAT!r}")
    for key in ("sample_rate", "lm_step", "lm_ema", "output_delay", "lm_config", "decoder_config"):
        if key not in meta:
            fail(f"{path}: __metadata__ lacks {key}")
    # Every tensor's byte range must lie inside the data section and match dtype x shape.
    size = os.path.getsize(path) - 8 - n
    width = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8}
    for tname, t in header.items():
        numel = 1
        for d in t["shape"]:
            numel *= d
        begin, end = t["data_offsets"]
        if end - begin != numel * width[t["dtype"]] or end > size or begin < 0:
            fail(f"{path}: tensor {tname} has an inconsistent byte range")
    return meta, header


def npy_header(path):
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            fail(f"{path}: not a .npy file")
        major, _minor = f.read(1)[0], f.read(1)[0]
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
            start = 10
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
            start = 12
        d = ast.literal_eval(f.read(hlen).decode("latin-1"))
    return d, start + hlen


files = {}
def record(rel):
    p = os.path.join(dest, rel)
    files[rel] = {"size": os.path.getsize(p), "sha256": sha256(p)}

fp32 = f"{name}.safetensors"
fp16 = f"{name}.fp16.safetensors"
meta32, t32 = safetensors_header(os.path.join(dest, fp32))
meta16, t16 = safetensors_header(os.path.join(dest, fp16))
if meta32 != meta16:
    fail("fp32 and fp16 files carry different __metadata__")
if set(t32) != set(t16):
    fail("fp32 and fp16 files carry different tensor names")
bad = [k for k, v in t32.items() if v["dtype"] != "F32"]
if bad:
    fail(f"{fp32}: not F32: {bad[:5]}")
bad = [k for k, v in t16.items() if v["dtype"] != ("F16" if len(v["shape"]) == 2 else "F32")]
if bad:
    fail(f"{fp16}: dtype policy violated (2-D must be F16, rest F32): {bad[:5]}")
for k in t32:
    if t32[k]["shape"] != t16[k]["shape"]:
        fail(f"tensor {k}: shape differs between fp32 and fp16 files")
n16 = sum(1 for v in t16.values() if v["dtype"] == "F16")
params = sum(__import__("math").prod(v["shape"]) for v in t32.values())
print(f"{fp32}: {len(t32)} tensors, all F32, {params / 1e6:.2f} M parameters")
print(f"{fp16}: {len(t16)} tensors, {n16} F16 matrices, {len(t16) - n16} F32")
print(f"metadata: format={meta16['format']} lm_step={meta16['lm_step']} lm_ema={meta16['lm_ema']} "
      f"sample_rate={meta16['sample_rate']} output_delay={meta16['output_delay']}")

voices = []
for fn in sorted(os.listdir(os.path.join(dest, "voices"))):
    rel = os.path.join("voices", fn)
    p = os.path.join(dest, rel)
    d, data_start = npy_header(p)
    shape = tuple(d.get("shape", ()))
    if d.get("descr") != "<f4" or d.get("fortran_order") is not False:
        fail(f"{rel}: expected little-endian float32 C-order, got {d}")
    if len(shape) != 2 or shape[1] != VOICE_DIM or shape[0] < 1:
        fail(f"{rel}: expected shape [T, {VOICE_DIM}], got {shape}")
    if os.path.getsize(p) != data_start + shape[0] * shape[1] * 4:
        fail(f"{rel}: file size does not match the header shape")
    voices.append({"file": rel, "frames": shape[0], "dim": shape[1]})
    print(f"{rel}: float32 [{shape[0]}, {shape[1]}]")

for rel in [fp16, fp32, "tokenizer.model"] + [v["file"] for v in voices] + ["WEIGHTS_LICENSE.md", "model-card.md"]:
    record(rel)

with open(os.path.join(dest, "SHA256SUMS"), "w") as f:
    for rel, info in files.items():
        f.write(f"{info['sha256']}  {rel}\n")

metadata = dict(meta16)
for key in ("lm_config", "decoder_config"):
    try:
        metadata[key] = json.loads(metadata[key])
    except (TypeError, ValueError):
        pass
manifest = {
    "name": name,
    "weights_version": version,
    "format": FORMAT,
    "date": date.today().isoformat(),
    "runtime_file": fp16,
    "reference_file": fp32,
    "tokenizer": "tokenizer.model",
    "tensors": len(t16),
    "parameters": params,
    "metadata": metadata,
    "voices": voices,
    "files": files,
}
with open(os.path.join(dest, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

(cd "$DEST" && sha256sum -c --quiet SHA256SUMS) && echo "SHA256SUMS verified"

if [ "$VERIFY" = 1 ]; then
  echo "== verify runtime"
  RUNTIME="$ROOT/runtime"
  (cd "$RUNTIME" && cargo build --release --example synth -p linnet-tts)
  SYNTH="$RUNTIME/target/release/examples/synth"
  RUNTIME_VERSION="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$RUNTIME/linnet-tts/Cargo.toml" | head -1)"
  VOICE="voices/$(basename "${voice_files[0]}")"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  "$SYNTH" "$DEST/$NAME.fp16.safetensors" "$DEST/tokenizer.model" "$DEST/$VOICE" "$TEXT" "$TMP/ref.wav" \
    "$TEMP" "$CFG" "$SEED" "$THREADS"
  MD5="$(md5sum "$TMP/ref.wav" | cut -d' ' -f1)"
  SAMPLES=$(( ($(stat -c %s "$TMP/ref.wav") - 44) / 2 ))
  echo "reference wav: $SAMPLES samples, md5 $MD5"
  python3 - "$DEST/MANIFEST.json" "$MD5" "$TEXT" "$VOICE" "$SEED" "$TEMP" "$CFG" "$THREADS" "$RUNTIME_VERSION" "$SAMPLES" <<'PY'
import json, sys
path, md5, text, voice, seed, temp, cfg, threads, rt, samples = sys.argv[1:]
m = json.load(open(path))
m["reference"] = {
    "reference_md5": md5,
    "wav": "16-bit PCM mono WAV written by runtime/linnet-tts/examples/synth.rs (44-byte header)",
    "samples": int(samples),
    "text": text,
    "voice": voice,
    "seed": int(seed),
    "temp": float(temp),
    "cfg": float(cfg),
    "threads": int(threads),
    "runtime_version": rt,
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PY
  echo "reference recorded in MANIFEST.json"
fi

echo "== done: $DEST"
ls -l "$DEST" "$DEST/voices"
