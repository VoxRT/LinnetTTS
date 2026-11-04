#!/usr/bin/env bash
# Set the release version in every package manifest of the repository.
#
#   build/set_version.sh <version>          e.g. build/set_version.sh 0.1.0
#
# Rewrites, and prints each file it changed:
#   runtime/linnet-ops/Cargo.toml, runtime/linnet-tts/Cargo.toml    Rust crates (the Linux C
#                                                                   tarball name is read from linnet-tts)
#   bindings/python/Cargo.toml, bindings/python/pyproject.toml      Python wheel `linnet-tts`
#   python/pyproject.toml, python/linnet_tts_torch/__init__.py      PyTorch package `linnet-tts-torch`
#   bindings/android/jni-crate/Cargo.toml                           Android JNI crate
#   bindings/android/gradle.properties (VERSION_NAME)               AAR com.voxrt.linnet:linnet-tts
#   Package.swift (let version)                                     SwiftPM binary target URL
#   bindings/ios/Sample/LinnetSample.xcodeproj/project.pbxproj      MARKETING_VERSION of the sample app
#   runtime/Cargo.lock, bindings/python/Cargo.lock,
#   bindings/android/jni-crate/Cargo.lock                           lock entries of the crates above
# The C API, the Python module, the Kotlin and the Swift wrappers report the version of the
# linnet-tts crate at run time.
set -euo pipefail

if [ $# -ne 1 ]; then sed -n '2,4p' "$0" >&2; exit 2; fi
V="$1"
if ! [[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.]+)?$ ]]; then
  echo "not a version: $V (expected MAJOR.MINOR.PATCH[-pre][+build])" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

changed=0
# rewrite <file> <sed-script>: apply in place, report when the content changed.
rewrite() {
  local f="$1" script="$2" before after
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
  before="$(sha256sum "$f")"
  sed -i -E "$script" "$f"
  after="$(sha256sum "$f")"
  if [ "$before" != "$after" ]; then echo "updated $f"; changed=$((changed + 1)); fi
}

# [package] / [project] tables: the first `version = "..."` line of the file.
for f in runtime/linnet-ops/Cargo.toml runtime/linnet-tts/Cargo.toml \
         bindings/python/Cargo.toml bindings/python/pyproject.toml \
         python/pyproject.toml bindings/android/jni-crate/Cargo.toml; do
  rewrite "$f" "0,/^version = \"[^\"]*\"/s//version = \"$V\"/"
done
rewrite python/linnet_tts_torch/__init__.py "s/^__version__ = \"[^\"]*\"/__version__ = \"$V\"/"
rewrite bindings/android/gradle.properties "s/^VERSION_NAME=.*/VERSION_NAME=$V/"
rewrite Package.swift "s/^let version = \"[^\"]*\"/let version = \"$V\"/"
rewrite bindings/ios/Sample/LinnetSample.xcodeproj/project.pbxproj "s/MARKETING_VERSION = [^;]*;/MARKETING_VERSION = $V;/"

# Cargo.lock: the version line that follows the name of one of our crates.
crates='linnet-ops|linnet-tts|linnet-tts-py|linnet-tts-jni'
for f in runtime/Cargo.lock bindings/python/Cargo.lock bindings/android/jni-crate/Cargo.lock; do
  rewrite "$f" "/^name = \"($crates)\"\$/{n;s/^version = \"[^\"]*\"/version = \"$V\"/}"
done

echo "$changed file(s) updated to $V"

# Every manifest must now agree.
bad=0
check() { grep -qE "$2" "$1" || { echo "mismatch in $1" >&2; bad=1; }; }
for f in runtime/linnet-ops/Cargo.toml runtime/linnet-tts/Cargo.toml bindings/python/Cargo.toml \
         bindings/python/pyproject.toml python/pyproject.toml bindings/android/jni-crate/Cargo.toml; do
  check "$f" "^version = \"$V\"\$"
done
check python/linnet_tts_torch/__init__.py "^__version__ = \"$V\"\$"
check bindings/android/gradle.properties "^VERSION_NAME=$V\$"
check Package.swift "^let version = \"$V\"\$"
check bindings/ios/Sample/LinnetSample.xcodeproj/project.pbxproj "MARKETING_VERSION = $V;"
[ "$bad" = 0 ] || exit 1

# The lock files must still match their manifests (no network needed).
if command -v cargo >/dev/null 2>&1; then
  for m in runtime/Cargo.toml bindings/python/Cargo.toml bindings/android/jni-crate/Cargo.toml; do
    cargo metadata --manifest-path "$m" --locked --offline --format-version 1 >/dev/null
  done
  echo "lock files consistent"
fi
