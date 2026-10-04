package com.kev

/**
 * The shared-state pair as the pipeline uses it: the state once, then one call per question on its
 * branch only. The app runs the LiteRT pair ([KevPairDecider]); the JVM tests put the oracle's
 * hidden states at the branch's readout positions instead.
 */
interface PairRunner {
  /** Ls: [runState] takes `ids` and `valid` with Ls entries each. */
  val stateLength: Int

  /** Lq: [runQuestion] takes `ids` and `valid` with Lq entries each. */
  val questionLength: Int

  /** Runs `[state] + state tokens` (padded to Ls) and keeps the state for the next questions. */
  fun runState(ids: IntArray, valid: FloatArray)

  /**
   * Runs one branch (padded to Lq) after the last [runState] and returns `hidden`, Lq ×
   * [KevPointerHead.HIDDEN_SIZE] floats, row-major. The graph continues the positions after the
   * state's real tokens, so the hidden states are those of the causal row state + branch.
   */
  fun runQuestion(ids: IntArray, valid: FloatArray): FloatArray
}

/**
 * The pair file's contract (the conversion run's `export_sharedstate_Ls128_Lq64.json`).
 * - `state_prefill_<Ls>`: in `ids` int32 and `valid` float32, both `[1, Ls]`; out the 48
 *   [stateNames] tensors.
 * - `question_step_<Ls>_<Lq>`: in `ids` int32 and `valid` float32, both `[1, Lq]`, `state_valid`
 *   float32 `[1, Ls]` (the state call's `valid`) and the 48 tensors; out `hidden` float32 of shape
 *   `[1, Lq, 1024]`.
 *
 * LiteRT has no call that lists the tensors of a signature, so the names come from the contract and
 * the compile checks each one.
 */
object KevPairContract {
  /** Decoder layers of Kev-0.8B; every fourth one (3, 7, …, 23) is full attention. */
  const val LAYERS = 24

  const val IDS = "ids"
  const val VALID = "valid"
  const val STATE_VALID = "state_valid"
  const val HIDDEN = "hidden"

  fun stateSignature(shape: KevPairShape): String = "state_prefill_${shape.stateLength}"

  fun questionSignature(shape: KevPairShape): String =
    "question_step_${shape.stateLength}_${shape.questionLength}"

  /** Layer [layer] is a full-attention layer (the others are Gated DeltaNet layers). */
  fun isAttention(layer: Int): Boolean = layer % ATTENTION_EVERY == ATTENTION_EVERY - 1

  /**
   * The 48 state tensors in the contract's order: `gdn_state_<l>` and `conv_tail_<l>` of the 18
   * Gated DeltaNet layers, `k_<l>` and `v_<l>` of the 6 attention layers.
   */
  val stateNames: List<String> =
    (0 until LAYERS).flatMap { layer ->
      if (isAttention(layer)) listOf("k_$layer", "v_$layer")
      else listOf("gdn_state_$layer", "conv_tail_$layer")
    }

  /** The shape of state tensor [name] for a pair of [stateLength] state tokens. */
  fun stateShape(name: String, stateLength: Int): List<Int> =
    when {
      name.startsWith("gdn_state_") -> listOf(1, GDN_HEADS, GDN_KEY_DIM, GDN_VALUE_DIM)
      name.startsWith("conv_tail_") -> listOf(1, CONV_TAIL, CONV_CHANNELS)
      name.startsWith("k_") || name.startsWith("v_") ->
        listOf(1, KV_HEADS, stateLength, ATTENTION_HEAD_DIM)
      else -> throw IllegalArgumentException("$name is not a state tensor")
    }

  private const val ATTENTION_EVERY = 4
  private const val GDN_HEADS = 16
  private const val GDN_KEY_DIM = 128
  private const val GDN_VALUE_DIM = 128
  private const val CONV_TAIL = 3
  private const val CONV_CHANNELS = 6144
  private const val KV_HEADS = 2
  private const val ATTENTION_HEAD_DIM = 256
}
