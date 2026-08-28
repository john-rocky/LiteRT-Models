# 次セッション指示 — NPU op-factor カタログ: 残スレッド(2026-08-28 更新)

phase-9/10 完了。優先度 A(prefill pad 則・RMSNorm・GELU LLM 版・GEMV dtype)と
B5/B6/B8、C-0/C-9 は**全て計測済み** — 結論は `NPU_OP_FACTOR_REPORT.md` 実験 8–13 章、
機械可読 24 行は `~/code/litert-compat/data/perf_factors_staging/`、恒久則は
memory `hexagon-graph-style`。以下は残った糸だけ。

## 起動場所・プロトコル(不変)

`~/Downloads/depthanything-android`。venv `~/venvs/ltconv040dev`。計測は npubench
`#sweep`(`-e model <path> -e accel npu|gpu [-e iters N]`)、S26=RFGL80R6A6H、
N=50 中央値、thermal NONE ゲート、compile→冷却→計測。
**S26 は他セッションと衝突しうる** — 開始前に `pgrep -fl phase` と ListAgents を確認
(2026-08-28 に相互 force-stop で計測を潰し合った前科あり)。

## 残スレッド(優先順)

1. **zipformer padded rebuild(0.38–0.42x の最後の容疑)**: dtype(実験1)・op 粒度
   (実験11)は棄却済み。残るは awkward 長(796/398/199/100)。mod-128 へ pad した
   再ビルドで決着する。predictor: decoder 級で ±1=4〜5 倍(実験 8)。
2. **qwen3emb flip**: max-norm SafeRMS ×113 → 定数スケール化で +23% 見込み(実験 9)。
   実体は `/Volumes/HD-SGDA/archive/Downloads/meeting/qwen3emb-work/`(HD 要マウント)。
   rwkv7_step は TANH×12+手動 norm 残存だが plain TANH は 1.10x(実験 5)—
   **鎖の中の TANH か**を opscan 文脈で先に確認してから着手判断。
3. **memorize 残り ~196ms(op 350–601)**: prefix 切りは 380–549 で QNN invoke 不能
   (ホストでは動く)。次は truncate でなく **op stub 化**(疑い op を恒等差し替えして
   差分計測)。cut ラダーは phase9.log / phase10.log、道具は cut_prefix.py の派生で。
4. **Mali per-module pin 確認**: 2.2.0 では builtin GELU も align_corners=True resize も
   可(C-9、Pixel 8a 実測)。dinov2/tipsv2/whisper 各モジュールの pin 版(2.1.3/2.1.5/
   2.1.6)で同じ 2 チェックを通してから erf/pad 資産に統一。
5. **whisper pad バンドル出荷**: `~/Downloads/meeting/npubench-phase9-probes/probe_wh_pad1536.tflite`
   は bit-exact・NPU 2.00x・transcript 無変化。カード/モジュール反映はオーナーレビュー
   前提でドラフトから。アプリ側変更は「mel を 3072 列に 0-pad」のみ。
6. **#1190 / tensorflow#126296(別セッション管轄、提出済み)**: merge されたら
   gelu_assoc_matrix.py を再実行して係数先行形が fusion されるか確認 → memory の
   「即効薬」行を畳む。
7. staging 24 行の promote 判断はオーナー(schema 0.1-draft のまま)。

## 前提の再確認先(結論を再導出しない)

- 実験 1–13: `NPU_OP_FACTOR_REPORT.md`(生ログ npubench/factors/phase1–10.log)
- C-0: `npubench/factors/c0_ab_tanhop.log`(GELU approximate=True 38.5ms)
- 融合行列: `npubench/factors/gelu_assoc_matrix.{py,log}`
- プローブ 50 本+pad 版 whisper: `~/Downloads/meeting/npubench-phase9-probes/`
- ホスト等価性の生値: 旧 scratchpad 消滅に注意 — 恒久化済みの分だけを信じる
