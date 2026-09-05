---
name: litert-feature-integration
description: Add an on-device ML feature (background removal, depth, detection, …) to an Android app that already exists, from a LiteRT-Models integration recipe (INTEGRATION.md + recipe.json) - the one dependency, the one drop-in Kotlin file, the model file with its checksum, pre/post-processing, cancel and release, and the on-device check that proves the integration. Use when the ask is "add X to this app" rather than "build a demo app"; for converting a model, verifying it, or scaffolding a new app, use gpu-clean-conversion, on-device-verification, and compiled-model-app-scaffolding instead.
---

# LiteRT feature integration

An integration is done when three things hold, in this order:

1. the app builds with one new dependency and one new Kotlin file, and the model
   file is in place with the checksum the recipe records,
2. **the recipe's check passes on a connected device** — the same command, values
   inside the recorded tolerance,
3. the feature's lifecycle is wired into the host app: inference off the main
   thread, `cancel()` when the screen goes away, `close()` on teardown, and no
   model blob in git.

Scope: adding an already-verified model to an app that already exists. The model
itself — conversion, GPU cleanliness, parity with the source — is upstream of this
skill and is taken from the recipe as given. If no recipe exists for the model you
need, do not improvise one from a sample app: run `gpu-clean-conversion` and
`on-device-verification` first, or pick a model that has a recipe.

## Step 0: find the recipe and read `recipe.json` before the prose

Recipes live in LiteRT-Models as `<model>/INTEGRATION.md` (prose, for people) and
`<model>/recipe.json` (the same facts, for agents). Worked example:
[`ormbg/`](../../ormbg/INTEGRATION.md) — background removal, GPU, Pixel 8a.

Read these keys and stop on the first mismatch with the host app:

| Key | Decides |
|---|---|
| `task`, `model.license`, `model.base_model` | whether this model is the right one and may ship in this app |
| `runtime.maven`, `runtime.version` | the exact dependency line — copy it, do not "upgrade" it |
| `integrate.min_sdk` | must be ≤ the app's `minSdk` |
| `integrate.files`, `integrate.deps` | what to copy and what to add |
| `model.hf_repo`, `model.file`, `model.sha256`, `model.bytes` | what to download and how to know it is the right file |
| `verify.command`, `verify.expected` | the pass/fail gate |
| `devices[]`, `unverified[]` | what the numbers mean and where they do not apply |

## Step 1: dependency

- Add the runtime exactly as `runtime.maven:runtime.version` (Google Maven). The
  `org.tensorflow:tensorflow-lite*` coordinates are the pre-rename names of the same
  runtime — do not add them alongside; see the naming table in the LiteRT-Models
  README if the app still uses them, and `litert-compiled-model-migration` if the
  app has `Interpreter` code to move.
- If the model is bundled: `androidResources { noCompress += "tflite" }`, so the
  file stays memory-mappable inside the APK.
- `arm64-v8a` is what every recipe verified.

## Step 2: model file

- Download `model.file` from the `model.hf_repo` resolve URL, then check
  `sha256` **and** `bytes` against `recipe.json` before anything else. A truncated or
  wrong file is the most common "the model is broken".
- Follow the delivery pattern the recipe verified — bundled in `assets/`, or staged
  into the app's `filesDir` for models too large to ship in the APK. The drop-in
  exposes a loader for each; the recipe says which one its numbers were taken on.
- Add the file to `.gitignore`. Weights are never committed.

## Step 3: the drop-in file

- Copy every path in `integrate.files` into the app's package and change only the
  `package` line. Do not "adapt" pre- or post-processing: it is part of the model
  contract, and the recipe's numbers hold for that code only. Model-specific values
  (input size, normalization) live in the file, not in the screen.
- Expect this surface and nothing app-specific: a factory (`fromAssets` /
  `fromFile`), `process(...)`, `cancel()`, `close()`.

## Step 4: lifecycle in the host app

- One owner — a ViewModel or a repository — creates the object once, on a
  background dispatcher, and calls `process` on the same confined dispatcher.
  `cancel()` when the screen leaves; `close()` in `onCleared` / `onDestroy`.
- Never on the main thread. Creation includes GPU shader compilation (seconds, not
  milliseconds; the recipe records `load_ms`); `process` is tens of milliseconds.
- Surface a GPU compile failure as an error. With `Accelerator.GPU` the whole graph
  compiles for the GPU or creation throws; there is no partial delegation to hide,
  and a CPU fallback you add yourself turns a 10× slowdown into a "working" app.

## Step 5: run the check

- Copy the recipe's instrumented test and its fixture into `src/androidTest`, run
  `verify.command` on a connected device, and compare the `RESULT` line in logcat
  with `verify.expected`. The tolerance in the test covers fp16 GPU noise; a value
  outside it is a wrong file, a changed fixture, or a preprocessing edit — in that
  order of likelihood.
- If your device is not in `devices[]`, you just measured a new row: record device,
  OS, date, `ms_per_frame`, and thermal status the way the recipe does.

## Watch for

- **Emulators do not run `CompiledModel` GPU.** The check is device-only; every
  recipe lists the emulator under `unverified`.
- **Latency moves with thermal status.** Compare numbers only at the status the
  recipe recorded; the test prints the status before and after.
- **A changed fixture invalidates `verify.expected`.** Re-record from the `RESULT`
  line; do not loosen the tolerance instead.
- **`run()` is asynchronous.** The drop-in already times run + readback together;
  if you add your own benchmark, do the same.

## Output

The diff of a finished integration is small and countable: one dependency line,
one `noCompress` line, one Kotlin file, one test plus fixture, one `.gitignore`
line, the model file absent from git — and the `RESULT` line from the device pasted
into the pull request.
