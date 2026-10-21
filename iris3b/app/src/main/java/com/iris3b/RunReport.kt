package com.iris3b

/**
 * Sums of one image's timeline ([StageLog] events), cut the way the engine's measurement entry
 * (BenchActivity) cuts them: graph loads and runs per part, and the host work between the graphs.
 */
object RunReport {

    /** Milliseconds per part: text encoder, adapter and DiT load / run, graph closes, host work. */
    fun stageMs(events: List<StageLog.Event>): Map<String, Double> {
        fun sum(stage: String, prefix: String? = null) = events
            .filter { e -> e.stage == stage && (prefix == null || (e.graph ?: "").startsWith(prefix)) }
            .sumOf { it.ms["ms"] ?: 0.0 }
        return linkedMapOf(
            "te_load" to sum("load", "te"),
            "te_run" to sum("run", "te"),
            "adapter_load" to sum("load", "dit_adapter"),
            "adapter_run" to sum("run", "dit_adapter"),
            "dit_load" to sum("load", "dit_") - sum("load", "dit_adapter"),
            "dit_run" to sum("run", "dit_") - sum("run", "dit_adapter"),
            "close" to sum("close"),
            "host" to events.filter { it.graph == null }.sumOf { it.ms["ms"] ?: 0.0 },
        )
    }

    /** The largest reading of memory field [key] (kB) across the timeline, or null. */
    fun peak(events: List<StageLog.Event>, key: String): Long? = events.mapNotNull { it.memory[key] }.maxOrNull()

    /** The timeline as JSON-ready maps. */
    fun events(events: List<StageLog.Event>): List<Map<String, Any?>> = events.map { e ->
        linkedMapOf("stage" to e.stage, "graph" to e.graph, "step" to e.step, "ms" to e.ms, "mem" to e.memory)
    }
}
