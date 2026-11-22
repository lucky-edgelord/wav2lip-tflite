# Wav2Lip accelerator bench (Android)

A small APK that runs one real Wav2Lip frame through every way LiteRT can reach a phone's
hardware, and reports **ms/frame** plus whether the output still **follows the audio** (a speech
mel window vs a silent one; the browser's WebGPU path silently produced a closed mouth).

| path | model | notes |
|---|---|---|
| CPU XNNPACK, 1 and 4 threads | full INT8 / INT8 weights | baseline |
| NNAPI (no NNAPI-CPU fallback) | full INT8; float with FP16 | Android's vendor-driver route to the DSP/NPU on older chips |
| LiteRT GPU delegate, FP16 and FP32 | INT8 weights | OpenCL/OpenGL, not WebGPU |
| Qualcomm QNN, DSP backend | full INT8 | Hexagon v66 DSPs (e.g. Snapdragon 680) |
| Qualcomm QNN, GPU backend | INT8 weights | Adreno via Qualcomm's own runtime |

Not included: QNN **HTP** and LiteRT's **NPU** path. Both need Snapdragon 8 Gen 1 (SM8450) or newer;
the HTP libraries alone are ~180 MB.

## Install

Download the APK from the repo's **Releases** (built by `.github/workflows/android.yml` on every
change under `android/`), allow installs from your browser, and open it. Tap **Run benchmark**,
then **Share** or **Copy** the results. Builds are signed with a throwaway debug key, so if an
install fails with a signature conflict, uninstall the previous version first.

## Build locally

```bash
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties   # Android SDK with platform 35
./gradlew assembleRelease                          # app/build/outputs/apk/release/app-release.apk
```

Models in `app/src/main/assets/` come from `scripts/convert.py --batch 1 --precision int8w int8
--augment-calib`; the `.f32` files are one real face crop + mel window and the PyTorch outputs
for speech and for silence.
