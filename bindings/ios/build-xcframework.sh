#!/usr/bin/env bash
# Build LinnetTTS.xcframework from the Rust runtime (runtime/linnet-tts, crate-type staticlib).
#
# Run on a Mac with Xcode and rustup targets:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios aarch64-apple-darwin
#   bindings/ios/build-xcframework.sh
#
# Slices: ios-arm64 (device), ios-arm64_x86_64-simulator (lipo of both simulator targets) and
# macos-arm64 (lets the Swift package build and test on an Apple-silicon host; drop it with
# --no-macos). Output: bindings/ios/dist/LinnetTTS.xcframework, a zip of it with the SwiftPM
# checksum, and a copy of the xcframework inside the Swift package
# (bindings/ios/LinnetTTS/LinnetTTS.xcframework) for the local binary target.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dist="$here/dist"
pkg="$here/LinnetTTS"
name="LinnetTTS"
with_macos=1
for a in "$@"; do
  case "$a" in
    --no-macos) with_macos=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

export PATH="$HOME/.cargo/bin:$PATH"
export IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-15.0}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"

# Keep build-host paths (home directory, cargo registry, checkout) out of the binaries' panic-location strings.
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$HOME=/home --remap-path-prefix=$(pwd)=/src"

targets=(aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios)
if [ "$with_macos" = 1 ]; then targets+=(aarch64-apple-darwin); fi

for t in "${targets[@]}"; do
  echo "== cargo build --release --target $t"
  cargo build --release --target "$t" -p linnet-tts --manifest-path "$root/runtime/Cargo.toml"
done

out="$root/runtime/target"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/headers" "$tmp/sim"
cp "$root/runtime/linnet-tts/include/linnet_tts.h" "$tmp/headers/"
cat > "$tmp/headers/module.modulemap" <<'MAP'
module CLinnetTTS {
    header "linnet_tts.h"
    export *
}
MAP

# Drop debug and local symbols (about -30 %); the exported linnet_* symbols stay. The app
# linker only pulls what it uses anyway: the runtime adds about 2 MB to an arm64 binary.
mkdir -p "$tmp/ios" "$tmp/macos"
cp "$out/aarch64-apple-ios/release/liblinnet_tts.a" "$tmp/ios/liblinnet_tts.a"
lipo -create \
  "$out/aarch64-apple-ios-sim/release/liblinnet_tts.a" \
  "$out/x86_64-apple-ios/release/liblinnet_tts.a" \
  -output "$tmp/sim/liblinnet_tts.a"
strip -S -x "$tmp/ios/liblinnet_tts.a" "$tmp/sim/liblinnet_tts.a"

args=(-library "$tmp/ios/liblinnet_tts.a" -headers "$tmp/headers"
      -library "$tmp/sim/liblinnet_tts.a" -headers "$tmp/headers")
if [ "$with_macos" = 1 ]; then
  cp "$out/aarch64-apple-darwin/release/liblinnet_tts.a" "$tmp/macos/liblinnet_tts.a"
  strip -S -x "$tmp/macos/liblinnet_tts.a"
  args+=(-library "$tmp/macos/liblinnet_tts.a" -headers "$tmp/headers")
fi

mkdir -p "$dist"
rm -rf "$dist/$name.xcframework" "$dist/$name.xcframework.zip" "$dist/$name.xcframework.zip.sha256" "$dist/$name.xcframework.checksum"
xcodebuild -create-xcframework "${args[@]}" -output "$dist/$name.xcframework"

# Zip for a GitHub release and the checksum SwiftPM wants for a URL binary target.
(cd "$dist" && ditto -c -k --keepParent "$name.xcframework" "$name.xcframework.zip")
(cd "$dist" && shasum -a 256 "$name.xcframework.zip" > "$name.xcframework.zip.sha256")
(cd "$pkg" && swift package compute-checksum "$dist/$name.xcframework.zip" > "$dist/$name.xcframework.checksum")

# Local copy for the Swift package's binary target (path: "LinnetTTS.xcframework").
rm -rf "$pkg/$name.xcframework"
cp -R "$dist/$name.xcframework" "$pkg/$name.xcframework"

echo
echo "== $dist/$name.xcframework"
for lib in "$dist/$name.xcframework"/*/liblinnet_tts.a; do
  printf '  %-34s %8s  %s\n' "$(basename "$(dirname "$lib")")" "$(du -k "$lib" | cut -f1)K" "$(lipo -archs "$lib")"
done
echo "  zip: $(du -k "$dist/$name.xcframework.zip" | cut -f1)K  SwiftPM checksum: $(cat "$dist/$name.xcframework.checksum")"
