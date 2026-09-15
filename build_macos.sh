#!/bin/bash
# Build the macOS .app + .dmg for distribution.
#
# Apple Silicon rules for a DMG handed to M1/M2/M3 users:
#   - The .app INSIDE the dmg must be arm64 (or the user needs Rosetta).
#     The bundle arch = THIS machine's arch — build the M1 DMG on an
#     Apple Silicon Mac.
#   - arm64 bundles MUST be signed (ad-hoc at minimum) or macOS refuses
#     to launch them.  This script always ad-hoc signs and verifies.
#   - For smoother distribution (no right-click->Open on first launch)
#     sign with a Developer ID and notarize:
#       CODESIGN_IDENTITY="Developer ID Application: ..." \
#       APPLE_ID=you@example.com APP_PASSWORD=xxxx \
#       ./build_macos.sh
set -e
cd "$(dirname "$0")"

echo "== host arch: $(uname -m) =="
[ "$(uname -m)" = "arm64" ] && echo "   -> this DMG will run natively on Apple Silicon"
[ "$(uname -m)" = "x86_64" ] && echo "   -> WARNING: x86_64 DMG — M1/M2 users will need Rosetta (macOS auto-prompts)"

python -m PyInstaller TFLiteTraining.spec --noconfirm

APP="dist/TFLiteTraining.app"
[ -d "$APP" ] || { echo "ERROR: bundle not found at $APP"; exit 1; }

echo "== bundle arch =="
file "$APP/Contents/MacOS/TFLiteTraining"

echo "== signing =="
if [ -n "${CODESIGN_IDENTITY:-}" ]; then
  codesign --force --deep --options runtime --sign "$CODESIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP" && echo "signature OK"

if [ -n "${APPLE_ID:-}" ] && [ -n "${APP_PASSWORD:-}" ]; then
  echo "== notarizing =="
  ditto -c -k --keepParent "$APP" /tmp/TFLiteTraining_notarize.zip
  xcrun notarytool submit /tmp/TFLiteTraining_notarize.zip \
    --apple-id "$APPLE_ID" --password "$APP_PASSWORD" --team-id "${APPLE_TEAM_ID:-}" --wait
  xcrun stapler staple "$APP" && echo "notarization OK"
  rm -f /tmp/TFLiteTraining_notarize.zip
else
  echo "== notarization skipped (set APPLE_ID/APP_PASSWORD to notarize) =="
fi

echo "== building dmg =="
python -m dmgbuild -s dmg_settings.py "TFLiteTraining" "dist/TFLiteTraining.dmg"
echo "== done: dist/TFLiteTraining.dmg =="
