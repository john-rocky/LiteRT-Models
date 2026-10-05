package com.kev

/** How a request runs: one row per question, or the state once and each question on the pair. */
enum class KevForm(val wireName: String) {
  ROW("row"),
  PAIR("pair"),
}

/** Which form a launch asks for; [wireName] is the `graph` extra's value. */
enum class KevGraphMode(val wireName: String) {
  /** The form with the smaller predicted time ([KevPlanner]). */
  AUTO("auto"),

  /** Rows only. */
  ROWS("rows"),

  /** The pair only; a request it cannot take fails with the reason. */
  PAIR("pair");

  companion object {
    fun of(name: String?): KevGraphMode? =
      if (name == null) AUTO else entries.firstOrNull { it.wireName == name.trim().lowercase() }
  }
}

/**
 * Whether a pair holds one copy of its weights for both signatures on the GPU (constant tensor
 * sharing); [wireName] is the `share` extra's value. Without sharing the pair is faster and needs
 * more memory (see [KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES]).
 */
enum class KevPairShare(val wireName: String) {
  /**
   * Without sharing when the phone has at least
   * [KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES] available right before the compile, else
   * with it.
   */
  AUTO("auto"),
  ON("on"),
  OFF("off");

  /** Whether a pair compiled with [availableBytes] available shares its weights. */
  fun sharesAt(availableBytes: Long): Boolean =
    when (this) {
      ON -> true
      OFF -> false
      AUTO -> availableBytes < KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES
    }

  companion object {
    /** The `share` extra: `auto`, `on` or `off`; null for anything else. */
    fun of(name: String): KevPairShare? = entries.firstOrNull {
      it.wireName == name.trim().lowercase()
    }
  }
}

/** One question step and one state call of a pair, in milliseconds. */
class KevPairCost(val stateMs: Double, val questionMs: Double)

/**
 * Milliseconds per graph call that [KevPlanner] compares: one question on each row window, and the
 * state and one question on each pair, with constant tensor sharing ([pairMs]) and without it
 * ([unsharedPairMs]).
 */
class KevCosts(
  val rowMs: Map<Int, Double>,
  val pairMs: Map<KevPairShape, KevPairCost>,
  val unsharedPairMs: Map<KevPairShape, KevPairCost> = pairMs,
) {
  companion object {
    /**
     * Galaxy S26 measurements of this app (GPU FP16_WITH_FP32_ACCUM, input writes + run +
     * read-back, cool starts, the median of the calls while the GPU clock ceiling stayed at its
     * 1,300 MHz), used only to choose a plan: one question on each window (L64 56.6 ms, L128 102.2,
     * L256 196.3, L512 388.8, L1024 816.0, L2048 1,803.5) and each pair's state call and question
     * step (Ls128 in the five-question request: 152.3 / 92.9 ms shared, 116.0 / 62.0 ms not shared;
     * Ls256 in a three-question request: 255.4 / 95.1 and 216.5 / 62.5 ms). One table for every
     * precision: a launch that forces FP32 plans with it too (an approximation).
     */
    val GALAXY_S26_GPU =
      KevCosts(
        rowMs =
          mapOf(
            64 to 56.6,
            128 to 102.2,
            256 to 196.3,
            512 to 388.8,
            1024 to 816.0,
            2048 to 1803.5,
          ),
        pairMs =
          mapOf(
            KevPairShape(128, 64) to KevPairCost(stateMs = 152.3, questionMs = 92.9),
            KevPairShape(256, 64) to KevPairCost(stateMs = 255.4, questionMs = 95.1),
          ),
        unsharedPairMs =
          mapOf(
            KevPairShape(128, 64) to KevPairCost(stateMs = 116.0, questionMs = 62.0),
            KevPairShape(256, 64) to KevPairCost(stateMs = 216.5, questionMs = 62.5),
          ),
      )
  }
}

/**
 * The predicted request time of each form, or null when that form cannot take the request;
 * [pairShared] says which pair costs [pairMs] used (null without a pair).
 */
class KevPrediction(val rowsMs: Double?, val pairMs: Double?, val pairShared: Boolean? = null)

/** Why the pair cannot take a request. */
sealed interface KevPairMiss {
  /** No pair file is installed. */
  data object NotInstalled : KevPairMiss

  /** `[state] + state tokens` is [tokens] long; the largest installed pair holds [window]. */
  data class StateTooLong(val tokens: Int, val window: Int) : KevPairMiss

  /** Question [index]'s branch is [tokens] long; the pair holds [window]. */
  data class QuestionTooLong(val index: Int, val tokens: Int, val window: Int) : KevPairMiss
}

/** How a request will run, chosen before any graph is compiled or closed. */
sealed interface KevPlan {
  /** A plan that runs, with the prediction of both forms. */
  sealed interface Ready : KevPlan {
    val form: KevForm
    val prediction: KevPrediction
  }

  /** Each question on its row window ([windows]). */
  class Rows(val windows: KevWindowPlan.Ready, override val prediction: KevPrediction) : Ready {
    override val form: KevForm
      get() = KevForm.ROW
  }

  /** The state once, then each of the [questions] on the pair [shape]. */
  class Pair(
    val shape: KevPairShape,
    val questions: Int,
    override val prediction: KevPrediction,
  ) : Ready {
    override val form: KevForm
      get() = KevForm.PAIR
  }

  /** No form takes the request: a row has no installed window ([missing]). */
  class NoWindow(val missing: KevWindowPlan.Missing) : KevPlan

  /** The pair was asked for and cannot take the request ([miss]). */
  class NoPair(val miss: KevPairMiss) : KevPlan
}

/**
 * Chooses how a request runs. The row form runs each question on the window it is assigned (the
 * smallest installed one that holds its row, see [KevResidentGraphs.assign]); the pair form runs
 * `[state] + state tokens` once and then each question's branch, so it takes a request only when
 * the state fits Ls and every branch fits Lq. With [KevGraphMode.AUTO] the form with the smaller
 * predicted time wins ([KevCosts], the row form on a tie); when only one form takes the request,
 * that one. Android-free.
 */
object KevPlanner {
  /**
   * The plan of a request whose state part is [stateTokens] long (`[state]` included) and whose
   * branches are [branchTokens] long, given the installed windows and pairs. [residentWindows] and
   * [availableBytes] decide whether two row windows may stay compiled
   * ([KevResidentGraphs.secondAllowed]); [fixedWindow] runs every row on that one window. A pair is
   * predicted with the costs of the way it would run: the resident pair ([residentPair]) as it was
   * compiled ([residentPairShared]), another one as [share] decides with [availableBytes] (an
   * approximation, like the second window: the compile reads the memory again).
   */
  fun plan(
    stateTokens: Int,
    branchTokens: List<Int>,
    installedWindows: List<Int>,
    installedPairs: List<KevPairShape>,
    residentWindows: List<Int>,
    availableBytes: Long,
    mode: KevGraphMode = KevGraphMode.AUTO,
    fixedWindow: Int? = null,
    costs: KevCosts = KevCosts.GALAXY_S26_GPU,
    share: KevPairShare = KevPairShare.AUTO,
    residentPair: KevPairShape? = null,
    residentPairShared: Boolean? = null,
  ): KevPlan {
    require(branchTokens.isNotEmpty()) { "A request has at least one question" }
    val rows = branchTokens.map { stateTokens + it }
    val windowPlan =
      if (fixedWindow != null) {
        KevResidentGraphs.planFixed(rows, fixedWindow, installedWindows)
      } else {
        KevResidentGraphs.plan(rows, installedWindows)
      }
    val rowsMs =
      (windowPlan as? KevWindowPlan.Ready)?.let { ready ->
        val second = KevResidentGraphs.secondAllowed(ready, residentWindows, availableBytes)
        KevResidentGraphs.assign(ready, second).sumOf { costs.rowMs.getValue(it) }
      }
    val pairs = installedPairs.filter {
      stateTokens <= it.stateLength && branchTokens.all { tokens -> tokens <= it.questionLength }
    }
    fun shared(shape: KevPairShape): Boolean =
      if (shape == residentPair && residentPairShared != null) residentPairShared
      else share.sharesAt(availableBytes)
    val pair = pairs.minByOrNull { pairMs(costs, it, branchTokens.size, shared(it)) }
    val prediction =
      KevPrediction(
        rowsMs,
        pair?.let { pairMs(costs, it, branchTokens.size, shared(it)) },
        pair?.let { shared(it) },
      )
    val rowPlan = (windowPlan as? KevWindowPlan.Ready)?.let { KevPlan.Rows(it, prediction) }
    val pairPlan = pair?.let { KevPlan.Pair(it, branchTokens.size, prediction) }
    if (mode == KevGraphMode.PAIR) {
      return pairPlan ?: KevPlan.NoPair(pairMiss(stateTokens, branchTokens, installedPairs))
    }
    if (mode == KevGraphMode.ROWS || fixedWindow != null || pairPlan == null) {
      return rowPlan ?: KevPlan.NoWindow(windowPlan as KevWindowPlan.Missing)
    }
    if (rowPlan == null) return pairPlan
    return if (requireNotNull(prediction.pairMs) < requireNotNull(prediction.rowsMs)) pairPlan
    else rowPlan
  }

  private fun pairMs(
    costs: KevCosts,
    shape: KevPairShape,
    questions: Int,
    shared: Boolean,
  ): Double {
    val cost = (if (shared) costs.pairMs else costs.unsharedPairMs).getValue(shape)
    return cost.stateMs + questions * cost.questionMs
  }

  /** Why no installed pair takes the request (the reason for the largest pair). */
  private fun pairMiss(
    stateTokens: Int,
    branchTokens: List<Int>,
    installedPairs: List<KevPairShape>,
  ): KevPairMiss {
    val largest =
      installedPairs.maxWithOrNull(compareBy({ it.stateLength }, { it.questionLength }))
        ?: return KevPairMiss.NotInstalled
    if (stateTokens > largest.stateLength) {
      return KevPairMiss.StateTooLong(stateTokens, largest.stateLength)
    }
    val index = branchTokens.indexOfFirst { it > largest.questionLength }
    return KevPairMiss.QuestionTooLong(index, branchTokens[index], largest.questionLength)
  }
}
