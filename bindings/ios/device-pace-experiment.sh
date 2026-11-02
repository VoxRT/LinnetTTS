#!/usr/bin/env bash
# Rebuilds and installs the sample on a device, then speaks one sentence twice with playback pacing
# and twice without it (-nopace), printing the per-step timings. Run on the Mac with an unlocked
# login keychain:
#   LINNET_DEVICE=<udid> DEVELOPMENT_TEAM=<team> bindings/ios/device-pace-experiment.sh [threads]
set -euo pipefail
: "${LINNET_DEVICE:?set LINNET_DEVICE to the device UDID}"
: "${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM}"
threads="${1:-1}"
here="$(cd "$(dirname "$0")" && pwd)"
dd="$HOME/Library/Developer/Xcode/DerivedData/linnet-sample-pace"
bundle=com.voxrt.linnet.sample
text="Honestly, I cannot schedule the tomato and herb advertisement for either route this quarter after the news at half past four."
"$here/build-xcframework.sh" >/dev/null
xcodebuild build -project "$here/Sample/LinnetSample.xcodeproj" -scheme LinnetSample -configuration Release \
  -destination "platform=iOS,id=$LINNET_DEVICE" -derivedDataPath "$dd" -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" 2>&1 | grep -E "BUILD|error:" || true
xcrun devicectl device install app --device "$LINNET_DEVICE" "$dd/Build/Products/Release-iphoneos/LinnetSample.app" | grep -i -E "installed|error" || true
for mode in "-noaudio" "-nopace -quiet" "-nopace" ""; do
  echo "== $mode, threads $threads"
  xcrun devicectl device process launch --console --terminate-existing --device "$LINNET_DEVICE" "$bundle" -- \
    -autospeak "$text" -threads "$threads" -repeat 1 $mode 2>&1 | grep -E "^\[linnet-sample\] qos" || true
done
