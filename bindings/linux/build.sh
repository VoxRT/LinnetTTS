#!/usr/bin/env bash
# Build the LinnetTTS C distribution for Linux.
#
#   bindings/linux/build.sh [--target <triple>] [--out <dir>] [--no-strip]
#
# Produces dist/linnet-tts-<version>-linux-<arch>/ with
#   lib/liblinnet_tts.so   shared library
#   lib/liblinnet_tts.a    static library (link with -lpthread -ldl -lm)
#   include/linnet_tts.h   C header
#   README.md, LICENSE
# and a .tar.gz of that directory next to it.
#
# Targets:
#   x86_64-unknown-linux-gnu   (default) built with the host toolchain
#   aarch64-unknown-linux-gnu  cross-built with `cargo zigbuild --target aarch64-unknown-linux-gnu.2.17`
#                              (glibc 2.17 floor) when cargo-zigbuild and zig are installed; otherwise
#                              falls back to `cargo build` with aarch64-linux-gnu-gcc as the linker if that
#                              cross compiler is present (glibc floor = the cross sysroot's glibc).
set -euo pipefail

TARGET=x86_64-unknown-linux-gnu
STRIP=1
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --target=*) TARGET="${1#--target=}"; shift ;;
    --out) OUT="$2"; shift 2 ;;
    --no-strip) STRIP=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUNTIME="$ROOT/runtime"
CRATE="$RUNTIME/linnet-tts"
VERSION="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$CRATE/Cargo.toml" | head -1)"
[ -n "$VERSION" ] || { echo "could not read version from $CRATE/Cargo.toml" >&2; exit 1; }

case "$TARGET" in
  x86_64-unknown-linux-gnu)  ARCH=x86_64;  GLIBC_MIN=2.17 ;;
  aarch64-unknown-linux-gnu) ARCH=aarch64; GLIBC_MIN=2.17 ;;
  *) echo "unsupported target $TARGET (x86_64-unknown-linux-gnu | aarch64-unknown-linux-gnu)" >&2; exit 2 ;;
esac
HOST_ARCH="$(uname -m)"
OUT="${OUT:-$HERE/dist}"
PKG="linnet-tts-$VERSION-linux-$ARCH"
DEST="$OUT/$PKG"

have() { command -v "$1" >/dev/null 2>&1; }

if ! rustup target list --installed 2>/dev/null | grep -qx "$TARGET"; then
  echo "rust target $TARGET is not installed: rustup target add $TARGET" >&2
  exit 1
fi

echo "== linnet-tts $VERSION -> $TARGET"
cd "$RUNTIME"
# Release profile for distribution: fat LTO, one codegen unit. Panics stay
# "unwind" because the C API catches them and turns them into error codes.
export CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=unwind
if [ "$ARCH" = "$HOST_ARCH" ]; then
  # Native build. Do not pass -C target-cpu: the kernels select AVX2/NEON at run time.
  cargo build --release -p linnet-tts --target "$TARGET"
elif have cargo-zigbuild && (have zig || python3 -c 'import ziglang' 2>/dev/null); then
  echo "== cross build via cargo zigbuild (glibc $GLIBC_MIN)"
  cargo zigbuild --release -p linnet-tts --target "$TARGET.$GLIBC_MIN"
else
  CROSS_CC=""
  case "$ARCH" in
    aarch64) CROSS_CC=aarch64-linux-gnu-gcc ;;
    x86_64)  CROSS_CC=x86_64-linux-gnu-gcc ;;
  esac
  if have "$CROSS_CC"; then
    echo "== zig not found; cross build via $CROSS_CC (glibc floor = this cross sysroot, not $GLIBC_MIN)" >&2
    LINKER_VAR="CARGO_TARGET_$(echo "$TARGET" | tr 'a-z-' 'A-Z_')_LINKER"
    env "$LINKER_VAR=$CROSS_CC" cargo build --release -p linnet-tts --target "$TARGET"
  else
    cat >&2 <<MSG
cross build for $TARGET needs one of:
  * cargo-zigbuild + zig (recommended, pins glibc $GLIBC_MIN):
        cargo install cargo-zigbuild
        pip install ziglang        # or install zig from https://ziglang.org/download/
  * a GNU cross compiler as the linker: apt install gcc-$(echo "$ARCH" | tr _ -)-linux-gnu
then rerun: $0 --target $TARGET
MSG
    exit 1
  fi
fi

LIBDIR="$RUNTIME/target/$TARGET/release"
SO="$LIBDIR/liblinnet_tts.so"
A="$LIBDIR/liblinnet_tts.a"
[ -f "$SO" ] && [ -f "$A" ] || { echo "build did not produce $SO / $A" >&2; exit 1; }

rm -rf "$DEST"
mkdir -p "$DEST/lib" "$DEST/include"
cp "$SO" "$A" "$DEST/lib/"
cp "$CRATE/include/linnet_tts.h" "$DEST/include/"
cp "$ROOT/LICENSE" "$DEST/LICENSE"
sed "s/@VERSION@/$VERSION/g; s/@ARCH@/$ARCH/g; s/@TARGET@/$TARGET/g" "$HERE/README.dist.md" > "$DEST/README.md"
mkdir -p "$DEST/examples"
cp "$HERE/examples/synth.c" "$HERE/examples/stream.c" "$HERE/examples/wav.h" "$HERE/examples/Makefile" "$DEST/examples/"

if [ "$STRIP" = 1 ]; then
  STRIP_BIN=strip
  if [ "$ARCH" != "$HOST_ARCH" ]; then
    if have "${ARCH}-linux-gnu-strip"; then STRIP_BIN="${ARCH}-linux-gnu-strip";
    elif have llvm-strip; then STRIP_BIN=llvm-strip;
    else STRIP_BIN=""; echo "== no cross strip found; shipping unstripped" >&2; fi
  fi
  if [ -n "$STRIP_BIN" ]; then
    "$STRIP_BIN" --strip-unneeded "$DEST/lib/liblinnet_tts.so"
    "$STRIP_BIN" --strip-debug "$DEST/lib/liblinnet_tts.a"
  fi
fi

( cd "$OUT" && tar czf "$PKG.tar.gz" "$PKG" && sha256sum "$PKG.tar.gz" > "$PKG.tar.gz.sha256" )

echo "== $DEST"
ls -l "$DEST/lib" "$DEST/include"
if [ "$ARCH" = "$HOST_ARCH" ] && have objdump; then
  echo "== glibc symbol versions required by liblinnet_tts.so:"
  objdump -T "$DEST/lib/liblinnet_tts.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -3
fi
echo "== $OUT/$PKG.tar.gz"
