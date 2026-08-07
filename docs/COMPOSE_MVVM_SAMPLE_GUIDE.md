# litert-samples Compose + MVVM sample guide

Practical, battle-tested recipe for restructuring (or building) a `google-ai-edge/litert-samples`
Android sample into the canonical **Jetpack Compose + MVVM** shape the reviewers require.

This is a **SECOND-pass guide**, by the project's explicit two-phase policy (see `CLAUDE.md`):
while getting the model to run on device, scaffold the app however is fastest (one Activity,
hardcoded strings, a custom `View`). Structure constraints during debugging regress
problem-solving. **Then, before the PR, restructure to the shape below.** Everything here is
distilled from converting all 32 open sample apps this way (2026-07) and device-verifying each on
a Pixel 8a — including the traps that cost real rework.

Companion docs: `docs/LITERT_CONVERSION_GUIDE.md` (getting the model GPU-clean), `CLAUDE.md`
(repo conventions + the canonical-template-by-output-type table).

---

## 1. The reference is the law

Mirror **`compiled_model_api/image_segmentation/kotlin_cpu_gpu/android`** (the reviewer's named
reference). Pick the concrete template by **output type**:

| Output | Copy this sample |
|---|---|
| One result bitmap (segmentation, matting, depth, boxes-on-bitmap, sketch, restore…) | `dichotomous_segmentation/dis_kotlin_gpu` |
| Boxes drawn live over an image | `object_detection/d_fine_kotlin_gpu` (Compose `Canvas` overlay) |
| Text in → text/label/score out | `semantic_similarity/kotlin_cpu_gpu` |
| **Audio → playback / labels / timeline** | `text_to_speech` (matcha, AudioTrack playback) + `sound_event_detection` (PANNs, mic + RECORD_AUDIO) |

Calibration facts (do NOT "fix" these — the reference itself does them):
- The reference has **no** `statusBarsPadding` on its own Scaffold; only the zoo added it. Both are
  fine — don't flag its absence, and adding it is the safe default (targetSdk 35 is edge-to-edge).
- `AndroidView` appears only inside the camera/gallery screens; not a defect.
- **Zero hardcoded `Text("…")`.** Every user-visible string is a `stringResource`.
- Compose **Material 1** (`androidx.compose.material.*`), never material3.

---

## 2. Canonical file set (package `com.google.ai.edge.examples.<task>`)

```
MainActivity.kt        thin ComponentActivity: by viewModels { MainViewModel.getFactory(this) },
                       setContent { ApplicationTheme { <Task>Screen(...) } },
                       collectAsStateWithLifecycle(), gallery via
                       rememberLauncherForActivityResult(PickVisualMedia())  (image apps)
MainViewModel.kt       owns the helper; confines EVERY model call to
                       Dispatchers.Default.limitedParallelism(1); loads from filesDir with an
                       inline "model not found — run install_to_device.sh" errorMessage;
                       onCleared() closes the helper
UiState.kt             @Immutable data class
ImageUtils.kt          only the helpers this app calls (image apps)
view/<Task>Screen.kt   Scaffold + TopAppBar + status header + picker Button + result area
view/Theme.kt          ApplicationTheme (copy from a sibling)
view/Color.kt          darkBlue / teal (+ only palettes this app actually uses)
res/values/strings.xml,themes.xml,colors.xml   no hardcoded UI strings
gradle/libs.versions.toml + build.gradle.kts   version catalog, alias(libs.plugins.…)
```

Build pins that actually delegate on device: **LiteRT `2.1.5`**, AGP `8.9.1`, Kotlin `2.2.21`,
compileSdk `36`, minSdk `26`, targetSdk `35`, `arm64-v8a`, JVM 17. Copy the version catalog and
root `build.gradle.kts` from a sibling sample **verbatim**; swap only `namespace`/`applicationId`.

---

## 3. The rules that are non-negotiable

1. **Never touch the inference helper.** Every `.kt` that is not `MainActivity` — the model wrapper
   **and** DSP/IO infra (`AudioDecoder`, `Istft`, `Fbank`, tokenizers, palettes, ONNX wrappers…) —
   stays byte-identical. Move the old Activity's render/decode/post-processing math into the
   ViewModel **verbatim**; do not re-derive it, do not change a constant.
2. **A render-only custom `View`** (`FaceView`, `MatchView`, `CompareView`, `TimelineView`) is NOT
   a helper — port its `onDraw` math into a ViewModel function that annotates a `bitmap.copy(...)`
   with `android.graphics.Canvas`, then delete the View. The screen shows the bitmap with
   `Image(resultImage.asImageBitmap())`. Because you draw at native resolution, the old fit-scale
   and offsets collapse to identity (`s=1, ox=oy=0`) — the drawing math is unchanged.
3. **Preserve the warm-up pass.** If the old Activity ran the model once untimed before the timed
   run (`run(bundled, warm = true)`), the displayed `ms` excludes first-run cost — keep it. Only
   drop it if the branch is provably dead (e.g. it runs on `assets/test_image.jpg` and the app has
   **no** `assets/` dir). Always check whether the branch actually executes.
4. **Preserve exact user-visible text.** A status/result string the old app built with
   `String.format`/interpolation keeps the SAME rendered text — put the format string in
   `strings.xml` with positional args (`%1$d`, `%2$.1f`), `translatable="false"` for pure
   numeric/metric layouts. The constraint is *where the format string lives*, not *what it renders*.
5. **Model loading — keep the app's own mechanism.** filesDir + `install_to_device.sh` → check
   `File(filesDir, MODEL).exists()` first and surface the install hint. assets /
   `download_model.gradle` → keep it (wire the undercouch plugin through the catalog); no filesDir
   migration. Multi-file models: check **every** required file exists and name the first missing one.

---

## 4. The traps that cost rework (read before you write)

- ⭐⭐ **Image sizing.** `Image(contentScale = Fit, modifier = Modifier.fillMaxWidth())` inside a
  `verticalScroll` renders the bitmap at **native pixel size**: with no height constraint the row is
  exactly the bitmap's pixel height, so Fit's scale is `min(boxW/bmpW, 1.0) = 1.0` and a 256/512-px
  model output looks tiny. **Fix:** drop the `verticalScroll`, put result images in a plain `Column`
  and give each `Modifier.fillMaxWidth().weight(1f)`. A helper composable that draws an image then
  needs a `ColumnScope` receiver. A before/after pair = two `weight(1f)` images so both fit one
  screen. `aspectRatio(w/h)` also fills the width but a pair no longer fits — use it only for a
  single short strip (e.g. a timeline). Only a trailing text list may scroll, inside its own
  `Modifier.weight(1f).verticalScroll(...)`.
- ⭐ **Conversion path depth.** `install_to_device.sh` references `../conversion` or
  `../../conversion` depending on directory depth: `<module>/kotlin_cpu_gpu/android` needs
  `../../conversion`; `<module>/<variant>_kotlin_gpu/android` needs `../conversion`. **Resolve the
  relative path against the filesystem — never eyeball the `../` count.** (This broke 3 apps.)
- ⭐ **Stale copyright.** `conversion/*.py` sometimes carry `Copyright 2025 The Google AI Edge
  Authors` in a file that is new-in-PR — the gate hard-fails on it. Bump to the current year.
  (This is distinct from `Copyright 2025 Google LLC` template files, which stay 2025.)
- ⭐⭐ **Dropped dependency when copying build files.** Copying a sibling's `build.gradle.kts` /
  version catalog can silently drop a dependency the app needs (e.g. `onnxruntime-android` for a
  PyanNet/ONNX hybrid). **Diff the ORIGINAL app's `dependencies {}` before overwriting** and re-add
  any that the template lacks (add the alias to the catalog too).
- **Scope traps.** Some PRs live inside a Google dir or beside a sibling app
  (`background_removal/ormbg_kotlin_gpu` next to `.../kotlin_cpu_gpu`; `image_classification/…`
  next to Google's own). Pass the **deepest exclusive subtree** to the gate, and confirm
  `git status` shows nothing changed outside your app's dir before committing.
- **Two apps in one PR.** A PR can carry two Gradle projects (e.g. `pose_estimation/kotlin_cpu_gpu`
  + `pose_estimation/rtmpose/…`). Judge and gate **per app**, not per PR.

---

## 5. Audio specifics

References: **matcha** `text_to_speech/MainViewModel.kt` (AudioTrack playback: `MODE_STATIC`,
`track.write(...).play()`, `play(FloatArray)`) and **PANNs** `sound_event_detection`
(mic `AudioRecord` + RECORD_AUDIO via a `RequestPermission` launcher in the Screen).

- Move the **mic capture** and the **AudioTrack playback** code from the old Activity into the
  ViewModel **verbatim** (same sample rate, same `AudioSource` — `MIC`/`UNPROCESSED`, same buffer
  math, same `AudioFormat`). Mic + model + playback all run on the confined dispatcher.
- **Audio output is playback, not an image.** Keep the old app's exact Play buttons and labels
  ("Play original"/"Play reconstructed", "Noisy"/"Enhanced", one button per stem/speaker).
- **File pick** = `rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument())`
  launched with `arrayOf("audio/*", "video/*")` → the untouched decoder. **Not** `PickVisualMedia`.
- Add `<uses-permission android:name="android.permission.RECORD_AUDIO"/>` when the app records.
- **Streaming** (e.g. a real-time tuner): run a mic loop in the ViewModel that republishes `UiState`
  per frame, toggled by an `AtomicBoolean`; `onCleared()` clears the flag.
- **Timeline / multi-track viz**: draw onto a bitmap in the ViewModel (rule 2) and show it with
  `aspectRatio(w/h)` (single strip); move its color palette out of the deleted View into the
  ViewModel and reuse it for the play-button tints.

---

## 6. Definition of done (per app)

1. `./gradlew clean :app:assembleDebug` → BUILD SUCCESSFUL, no new warnings.
2. Pre-PR gate `pre_pr_check.sh <worktree> <module-subtree>` → **exit 0 / PASS**. (One accepted
   exception: the gate's `expand_braces_kt` may brace-expand a helper's single-statement bodies —
   verify the change is **brace-only** by comparing the token stream with braces/whitespace
   stripped, then accept; it matches the repo's own style bar.)
3. **Device-verify on a real device:** push the model via `install_to_device.sh`, launch, confirm
   GPU residency in logcat (`Replacing N out of N node(s) with delegate (LITERT_CL)`, filtered by
   the app's **pid** — other apps emit the same line), zero crashes, and eyeball the render. Drive
   the actual interaction (paint a mask, record, play a stem) for anything you hand-wrote. Compare
   against the pre-change binary when output correctness is in doubt (no-regression check).
4. Commit local, gate PASS, base == origin tip, **no model blobs in the commit**; push; then
   re-verify the remote SHA with `git ls-remote` (don't trust the push message).

---

## 7. Reusable tooling (this session)

- `~/Downloads/meeting/style-sweep-tools/pre_pr_check.sh` — the mandatory gate (format autofix +
  banned-pattern greps + 100-col + build + scope guard).
- Independent per-app checker (helper-untouched, no `setContentView`/material3, confined dispatcher,
  no hardcoded strings, scope) and a device-verify driver were kept in the session scratchpad; the
  patterns are described here so they can be re-created.
- Full inventory + per-app device evidence: `~/Downloads/meeting/compose-mvvm-audit.md`.
