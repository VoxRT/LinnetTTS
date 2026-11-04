#!/usr/bin/env bash
# Build liblinnet_tts_jni.so for the Android ABIs and drop them into the
# library module's jniLibs/ directory.
#
#   bindings/android/build-native.sh                 # arm64-v8a + x86_64
#   bindings/android/build-native.sh arm64-v8a       # one ABI
#   ABIS="arm64-v8a armeabi-v7a x86_64" bindings/android/build-native.sh
#
# Needs: an Android NDK (ANDROID_NDK_HOME, or ANDROID_HOME/ndk/<ver>),
# cargo-ndk (`cargo install cargo-ndk`) and the Rust targets
# (`rustup target add aarch64-linux-android x86_64-linux-android`).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
crate="$here/jni-crate"
out="$here/linnet-tts/src/main/jniLibs"
platform=${ANDROID_PLATFORM:-24}

if [ $# -gt 0 ]; then
  abis=("$@")
else
  read -r -a abis <<<"${ABIS:-arm64-v8a x86_64}"
fi

if ! command -v cargo-ndk >/dev/null; then
  echo "cargo-ndk not found: cargo install cargo-ndk" >&2
  exit 1
fi
if [ -z "${ANDROID_NDK_HOME:-}" ] && [ -z "${ANDROID_NDK_ROOT:-}" ]; then
  if [ -n "${ANDROID_HOME:-}" ] && [ -d "$ANDROID_HOME/ndk" ]; then
    ANDROID_NDK_HOME=$(ls -d "$ANDROID_HOME"/ndk/* | sort -V | tail -1)
    export ANDROID_NDK_HOME
  else
    echo "no NDK: set ANDROID_NDK_HOME (or install one under \$ANDROID_HOME/ndk)" >&2
    exit 1
  fi
fi

targets=()
for abi in "${abis[@]}"; do
  targets+=(-t "$abi")
done

# Keep build-host paths (home directory, cargo registry, checkout) out of the binaries' panic-location strings.
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$HOME=/home --remap-path-prefix=$(pwd)=/src"

echo "NDK: ${ANDROID_NDK_HOME:-$ANDROID_NDK_ROOT}, platform $platform, ABIs: ${abis[*]}"
(cd "$crate" && cargo ndk --platform "$platform" "${targets[@]}" -o "$out" build --release)

for abi in "${abis[@]}"; do
  # cargo-ndk also copies the runtime's own cdylib; the JNI library links
  # the runtime statically and does not need it.
  rm -f "$out/$abi/liblinnet_tts.so"
  so="$out/$abi/liblinnet_tts_jni.so"
  printf '%-12s %8d KB  %s\n' "$abi" "$(( $(stat -c %s "$so") / 1024 ))" "$(sha256sum "$so" | cut -c1-16)"
done
