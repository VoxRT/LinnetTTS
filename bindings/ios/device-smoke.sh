#!/usr/bin/env bash
# Exercise the iOS channel on an attached iPhone: run the hosted XCTests (LinnetSampleTests),
# then install the sample app, copy the model files into its Documents and launch it in
# autospeak mode, printing the timings it logs.
#
# Usage:
#   LINNET_DEVICE=<udid> [DEVELOPMENT_TEAM=<team>] bindings/ios/device-smoke.sh [tests|sample|all]
#
#   LINNET_DEVICE      device UDID (xcrun xctrace list devices); devicectl accepts it too
#   DEVELOPMENT_TEAM   signing team; may instead live in bindings/ios/local.xcconfig (untracked)
#   LINNET_ASSETS      directory with linnet-en-35m.fp16.safetensors, tokenizer.model and a
#                      voice .npy (default: <repo>/local)
#   LINNET_TEXT        sentence for the sample run
#
# Signing happens on this Mac, so the login keychain must be unlocked (an SSH session cannot
# unlock it: codesign then fails with errSecInternalComponent). On the first install the
# phone may ask to trust the developer: Settings > General > VPN & Device Management.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
what="${1:-all}"
: "${LINNET_DEVICE:?set LINNET_DEVICE to the device UDID}"
assets="${LINNET_ASSETS:-$root/local}"
text="${LINNET_TEXT:-Hello there. This is Linnet speaking from your phone, one step at a time.}"
dd="${LINNET_DERIVED_DATA:-$HOME/Library/Developer/Xcode/DerivedData/LinnetSample-device}"
bundle=com.voxrt.linnet.sample
team_arg=()
if [ -n "${DEVELOPMENT_TEAM:-}" ]; then team_arg=("DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM"); fi

voice="$(ls "$assets"/voice.npy "$assets"/voice_*.npy 2>/dev/null | head -1 || true)"
[ -n "$voice" ] || { echo "no voice .npy in $assets" >&2; exit 1; }
for f in linnet-en-35m.fp16.safetensors tokenizer.model; do
  [ -f "$assets/$f" ] || { echo "missing $assets/$f" >&2; exit 1; }
done

if [ "$what" = tests ] || [ "$what" = all ]; then
  echo "== hosted tests on $LINNET_DEVICE"
  tdir="$here/LinnetTTS/Tests/LinnetTTSTests/Assets"
  cp "$assets/linnet-en-35m.fp16.safetensors" "$assets/tokenizer.model" "$tdir/"
  cp "$voice" "$tdir/voice.npy"
  xcodebuild test -project "$here/Sample/LinnetSample.xcodeproj" -scheme LinnetSample \
    -destination "platform=iOS,id=$LINNET_DEVICE" -derivedDataPath "$dd" \
    -allowProvisioningUpdates -allowProvisioningDeviceRegistration "${team_arg[@]}" 2>&1 | tee "$dd.test.log" \
    | grep -E "\[linnet\]|Test Case.*(passed|failed|skipped)|error:|TEST .*(SUCCEEDED|FAILED)|Executed" \
    | sed 's/^.*\] \[linnet\]/[linnet]/' | sort -u
  rm -f "$tdir/linnet-en-35m.fp16.safetensors" "$tdir/tokenizer.model" "$tdir/voice.npy"
fi

if [ "$what" = sample ] || [ "$what" = all ]; then
  echo "== sample app on $LINNET_DEVICE"
  xcodebuild build -project "$here/Sample/LinnetSample.xcodeproj" -scheme LinnetSample -configuration Release \
    -destination "platform=iOS,id=$LINNET_DEVICE" -derivedDataPath "$dd" \
    -allowProvisioningUpdates -allowProvisioningDeviceRegistration "${team_arg[@]}" 2>&1 | grep -E "error:|BUILD" || true
  app="$dd/Build/Products/Release-iphoneos/LinnetSample.app"
  xcrun devicectl device install app --device "$LINNET_DEVICE" "$app"
  for f in "$assets/linnet-en-35m.fp16.safetensors" "$assets/tokenizer.model"; do
    xcrun devicectl device copy to --device "$LINNET_DEVICE" --source "$f" \
      --destination "Documents/$(basename "$f")" --domain-type appDataContainer --domain-identifier "$bundle"
  done
  xcrun devicectl device copy to --device "$LINNET_DEVICE" --source "$voice" \
    --destination "Documents/voice.npy" --domain-type appDataContainer --domain-identifier "$bundle"
  for threads in 1 0; do
    echo "-- autospeak, threads $threads"
    xcrun devicectl device process launch --device "$LINNET_DEVICE" --console --terminate-existing "$bundle" \
      -autospeak "$text" -threads "$threads" -repeat 3 2>&1 | grep -E "linnet-sample|error" || true
  done
fi
