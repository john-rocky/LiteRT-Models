#!/usr/bin/env bash
# Push the Nemotron-3-Diarization graphs into the app's private filesDir/models (too big for the APK).
# Install the app (./gradlew :app:installDebug), then:
#   ./install_to_device.sh <dir-with-the-tflite> [model.tflite ...]
# Default models: the streaming pair nemotron3_diar_frontend.tflite and
# nemotron3_diar_encoder_low_latency_fp16.tflite (add nemotron3_diar_encoder_offline_fp16.tflite for the
# offline file mode). Set ANDROID_SERIAL to pick the device.
set -e
PKG=com.nemotron3diar
DIR="${1:-.}"
shift || true
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(nemotron3_diar_frontend.tflite nemotron3_diar_encoder_low_latency_fp16.tflite)
fi
adb shell run-as $PKG mkdir -p files/models
for M in "${MODELS[@]}"; do
  echo "pushing $M ..."
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell chmod 644 "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/models/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/models/
echo "done"
