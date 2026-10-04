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

/** One question step and one state call of a pair, in milliseconds. */
class KevPairCost(val stateMs: Double, val questionMs: Double)

/**
 * Milliseconds per graph call that [KevPlanner] compares: one question on each row window, and the
 * state and one question on each pair.
 */
class KevCosts(val rowMs: Map<Int, Double>, val pairMs: Map<KevPairShape, KevPairCost>) {
  companion object {
    /**
     * Galaxy S26 measurements of this app (GPU FP32, input writes + run + read-back), used only to
     * choose a plan: one question on each window (L128 175.8 ms, L256 323.3, L512 615.1, L1024
     * 1,333.4 from cool starts; L2048 3,175 from a warm run) and the pair's state call and question
     * step in the five-question request from a cool start (271.2 ms, 185.3 ms).
     */
    val GALAXY_S26_GPU_FP32 =
      KevCosts(
        rowMs = mapOf(128 to 176.0, 256 to 323.0, 512 to 615.0, 1024 to 1333.0, 2048 to 3175.0),
        pairMs = mapOf(KevPairShape(128, 64) to KevPairCost(stateMs = 271.0, questionMs = 185.0)),
      )
  }
}

/** The predicted request time of each form, or null when that form cannot take the request. */
class KevPrediction(val rowsMs: Double?, val pairMs: Double?)

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
   * ([KevResidentGraphs.secondAllowed]); [fixedWindow] runs every row on that one window.
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
    costs: KevCosts = KevCosts.GALAXY_S26_GPU_FP32,
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
        val second =
          residentWindows.containsAll(ready.windows.toSet()) ||
            availableBytes >= KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES
        KevResidentGraphs.assign(ready, second).sumOf { costs.rowMs.getValue(it) }
      }
    val pairs = installedPairs.filter {
      stateTokens <= it.stateLength && branchTokens.all { tokens -> tokens <= it.questionLength }
    }
    val pair = pairs.minByOrNull { pairMs(costs, it, branchTokens.size) }
    val prediction = KevPrediction(rowsMs, pair?.let { pairMs(costs, it, branchTokens.size) })
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

  private fun pairMs(costs: KevCosts, shape: KevPairShape, questions: Int): Double {
    val cost = costs.pairMs.getValue(shape)
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
