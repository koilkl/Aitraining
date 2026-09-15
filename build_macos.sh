#!/bin/bash
# Build the macOS .app bundle.
#
# Apple Silicon (M1/M2/M3) rules:
#   - arm64 bundles MUST be ad-hoc signed or macOS refuses to launch them
#     ("app damaged" / instant quit / not responding).  PyInstaller >= 6
#     signs automatically; this script re-signs to be certain.
#   - The bundle architecture = the BUILD machine's architecture.  An
#     x86_64 bundle needs Rosetta on Apple Silicon; build on the target
#     arch machine for best results.
set -e
cd "$(dirname "$0")"

echo "== host arch: $(uname -m) =="

python -m PyInstaller TFLiteTraining.spec --noconfirm

APP="dist/TFLiteTraining.app"
[ -d "$APP" ] || { echo "ERROR: bundle not found at $APP"; exit 1; }

echo "== bundle arch =="
file "$APP/Contents/MacOS/TFLiteTraining"

echo "== ad-hoc signing =="
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "signature OK"

echo "== done: $APP =="
