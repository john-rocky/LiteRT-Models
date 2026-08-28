# 次セッション指示 — NPU op-factor カタログ拡張(LLM 適用+規則の一般化)

そのまま新セッションの最初のプロンプトとして貼れる形で書いてある。

---

## 起動場所

**`~/Downloads/depthanything-android` で立ち上げる**(litertlm-convert ではない)。
理由: hexagon-graph-style ほか今回の則の memory はこのプロジェクトにしか載らない。
道具(npubench / factors / probes)もレポートも rwkv7・qwen3 の実モデルもこちら。
litertlm-convert は A3(Gemma .litertlm 測定)と A4 の int4/int8 実物の所在確認で
**ファイルを読むだけ**。.litertlm レーンを本格的に掘る段になったら、その部分だけ
litertlm-convert で別セッションを立てる(あちらの memory に Portal / LiteRT-LM 手順がある)。

## 2026-08-25 更新 — Mac 側準備完了、残りは S26 計測のみ

S26 不在のまま進めた分。**S26 が空いたら最初にやること:**

1. C-0(3分): `ab_tanhop.tflite` の NPU sweep → issue #1190 に1コメント(下の C-0 参照)
2. phase-9 一括計測:
   `cd npubench/factors && nohup zsh phase9_chain.sh ~/Downloads/meeting/npubench-phase9-probes > phase9_nohup.out 2>&1 &`
   (push→全コンパイル→冷却ゲート 50-iter 計測。結果は probe dir の phase9.log)

**済んでいること:**
- プローブ 50 本を全て生成・ホスト検証済み → `~/Downloads/meeting/npubench-phase9-probes/`
  (batch4: pf 掃引16+RMS2、batch6: fx10+gr7、whisper ペア2、GEMV dtype 4、memorize cut 9。
  ビルドは `build_phase9_probes.sh`、ログは同 dir の build9.log)
- whisper pad1536 は **bit-exact**(max_abs 0.0)。素朴 pad は 2e-2 ずれる —
  修正2点(conv1 末尾マスク+checkpoint 保存 pos table 使用)は build_whisper_pad.py の
  docstring に記録。wh_ctrl は同一ビルダーの T=1500 再構築(shipped ファイルは端末上、
  「vs shipped」比較のアンカーは端末の実ファイルのまま)
- A2 の opscan 済み: rwkv7_step = TANH×12 + 手動 norm(SUM×48/RSQRT×50)残存、
  qwen3emb = max-norm SafeRMS ×113 → 両方 flip 候補として有効。
  RMS_NORM 組み込み op は schema に無し(2.1.6 / 2.3.0.dev 確認済、probe_batch4 docstring)
- memorize bisect の元ファイル: edgetam-video アセットが sweep 計測物と完全一致を確認(601 ops)

## 貼る指示文

前セッションで「数学的等価な書き換え → NPU 実測倍率」のカタログを 16 行作り、
DINOv2-S(2.05x)と TIPSv2-DPT(2.30x)を実モデル flip した。続きをやる。

**先に読む(結論を再導出しない):**
1. `NPU_OP_FACTOR_REPORT.md`(zoo repo 直下)— 実験 1–7 の全数値
2. `npubench/factors/` — プローブ生成・fold・計測スクリプトと phase*.log
3. `~/code/litert-compat/data/perf_factors_staging/` — 機械可読 16 行 + draft schema
4. memory `hexagon-graph-style` — 確立済みの則(dtype null / GELU 5.8x / 長さの数字 /
   融合文脈 / 逆転 2 件)

**プロトコル(前セッションで確立、逸脱しない):**
- venv は `~/venvs/ltconv040dev`(litert-torch 0.9.3 + ai-edge-litert 2.1.6)
- 計測は npubench `#sweep`(`-e model <path> -e accel npu|gpu [-e iters N]`)、
  S26 = RFGL80R6A6H、N=50 中央値、**thermal NONE ゲート必須**
  (`npubench/factors/gated_measure.sh` の wait_cool を流用。コンパイルだけで SEVERE まで
  加熱するので「全コンパイル→冷却→ゲート計測」の順)
- 長時間パイプラインは nohup + phase.log + Monitor(529 死に耐えた実績のある形)
- プローブは同一シード・matched-compute・1 変数。**GPU アームを取るなら rank≤4**
  (qkv permute の rank-5 は ML Drift が拒否)
- 等価性はホスト CPU で先に数値検証(bit-exact / 恒等 / 近似+corr を行に明記)
- refute-first。Mali 世代の互換則をこの環境に持ち込まない(逆転 2 件の前科)
- HF アップロードは `huggingface-cli`(`hf` シムは venv 欠損で壊れている 08-25 時点)

### 優先度 A — LLM 適用

1. **prefill 長 pad 則の確定(最安の果実)**: decoder 風ブロック(RMSNorm+SwiGLU+GQA、
   d=1024 級)で T ∈ {1152, 1280, 1408, 1536, 1664} と各±1、および 1500 を掃引。
   仮説「T ≡ 0 mod 128 が速い」を確定 or 棄却。確定すれば「プロンプトを 128 の倍数に
   pad すると NPU prefill ~2x」という出荷可能な一行になる。
2. **RMSNorm**: 手動分解(SafeRMS scale-before-square / max-norm 版 — memory
   `rtmpose` / `qwen3_embedding`)vs 組み込み RMS_NORM op(2.2.0 の schema に有無を先に確認)。
   factor が出たら実モデル flip 候補は qwen3 embedder と rwkv7_step
   (rwkv7 は既に 1.70x 勝ちだが op 構成を opscan して tanh/手動 norm が残っていれば上積み)。
3. **GELU flavor の LLM 版**: Gemma の公式活性は gelu_pytorch_tanh =例の 5.8x 形。
   CompiledModel 自前レーンに tanh 分解の LLM があれば flip。**.litertlm(LiteRT-LM)の
   グラフは Google 側なので書き換えず、測定・報告まで**(memory
   `llm_on_compiledmodel_gpu` の分界)。
4. **decode GEMV の dtype(逆転予測の検証)**: fp16-dtype null は compute-bound の結論。
   T=1 GEMV(重み 100–500MB 級、int8/int4/fp16/fp32)プローブで帯域律速領域を測る。
   「同じ書き換えでも領域で結論が逆転する」が示せればカタログの scope 欄の価値が立つ。

### 優先度 B — 規則の一般化(もっと掘る)

5. **融合破壊 op 一覧**: gelu 鎖の TANH→LOGISTIC で 2.4x 差が出た(=TANH が QNN の
   鎖融合を壊す)。同一鎖に 1 op だけ差し替えるプローブ族で EXP / POW / ERF? / ABS /
   MAXIMUM / RSQRT を判定し「融合安全 op リスト」を作る。
6. **op 粒度カーブ(zipformer 軸)**: 同 FLOPs を N 個の直列 op に分割(N=10..3000)して
   NPU 時間 vs N を測る。zipformer(3085 op、0.38–0.42x)の未説明分を定量化。
7. memorize(0.04x)bisect: 疑いは batch-256 [1,64]×[64,16] matmul 群と
   [1,64,64,1024] builtin GELU。プレフィックス切りで犯人区間を特定。
8. whisper 実モデルの 1536-pad flip: pos_embed 延長+attention mask で数学的に閉じた形に
   してから実測(probe 予測: 37.6→~15ms、GPU 26.6 に対し flip)。精度は WER で検証。

### 優先度 C — 出荷・整理

0. **【完了 2026-08-28 — 再実行不要】#1190 フォローアップ済み**: C-0 実測 38.476ms(c0_ab_tanhop.log)を issue にコメント投稿、fusion PR = tensorflow/tensorflow#126296。根因 matrix = gelu_assoc_matrix.py。以下は当時の手順(記録): S26 が空いたら
   `an/dinov2_ab_tanhop_fp32.tflite`(GELU approximate=True、tanh 鎖と数値同一)を
   npubench sweep で NPU 計測(thermal NONE ゲート)し、issue に1コメントで追記する。
   erf 版(41.9ms)と同等なら fusion 提案が完結、遅ければその事実を正直に書く。
   ファイルは ~/Downloads/meeting/dinov2-work/dinov2_ab_tanhop_fp32.tflite、かつ
   **S26 の /data/local/tmp/npubench/ab_tanhop.tflite に push 済み**(即計測可)。
   消えていれば build_dinov2_ab.py の gelu を `F.gelu(approximate='tanh')` にして再生成。


9. Pixel 8a(Mali)接続時: builtin GELU op と align_corners=True RESIZE_BILINEAR の
   compile 可否を各 1 回チェック → 可なら dinov2/tipsv2 のアセットを erf 版に統一できる。
   不可なら現状の「NPU レーン並行ファイル」を維持。
10. 【完了 2026-08-25】カード更新済み: DINOv2-ViT-S14-LiteRT に dinov2_s_erf_fp16.tflite、
    TIPSv2-B14-DPT-LiteRT に tipsv2_b14_dpt_erf_fp16.tflite を公開、両カードの NPU
    セクション書き換え済み。litert-torch issue #1190 起票済み(フォローアップは上の 0)。
11. `perf_factors_staging` の promote 判断はオーナー。新しい行は同じ schema で追記。

### 成果物の置き場(前セッションと同じ)
- 数値とログ → `npubench/factors/` に追記、レポートは `NPU_OP_FACTOR_REPORT.md` に章追加
- 機械可読行 → `litert-compat/data/perf_factors_staging/`
- 恒久則だけ memory `hexagon-graph-style` を上書き更新(state は書かない)
