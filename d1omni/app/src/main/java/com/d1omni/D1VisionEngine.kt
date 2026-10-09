package com.d1omni

import android.content.Context
import java.io.Closeable
import java.io.File

/**
 * The picture path's entries of `contract.json`, Android-free: the vision tower and the projector
 * (`graphs.vision_tower` / `graphs.projector`: file and signature; `files[]`: bytes, sha256 and the
 * inputs / output in tensor order) and the position table (`vision_position_table`). The shapes must
 * be the ones [D1Vision] lays out.
 */
class D1VisionContract(json: Any?) {
  /** One single-signature graph: its file entry and its float32 tensors. */
  class Graph(
    val file: String,
    val signature: String,
    val bytes: Long,
    val sha256: String,
    /** Input name -> shape, in tensor order. */
    val inputs: LinkedHashMap<String, List<Int>>,
    val output: Pair<String, List<Int>>,
  )

  val tower: Graph
  val projector: Graph
  val tableFile: String
  val tableSha256: String

  init {
    val root = json as Map<*, *>
    val graphs = root["graphs"] as Map<*, *>
    val files = (root["files"] as List<*>).map { it as Map<*, *> }.associateBy { it["name"] as String }
    fun graph(key: String): Graph {
      val entry = graphs[key] as Map<*, *>
      val name = entry["file"] as String
      val signature = entry["signature"] as String
      val file = requireNotNull(files[name]) { "contract.json lists no file $name" }
      require(file["signature"] == signature) { "contract.json: $name is not $signature" }
      fun tensors(list: Any?): List<Pair<String, List<Int>>> =
        (list as List<*>)
          .map { it as Map<*, *> }
          .sortedBy { (it["tensor_index"] as JsonNumber?)?.toInt() ?: 0 }
          .map { tensor ->
            val tensorName = tensor["name"] as String
            require(tensor["dtype"] == "FLOAT32") { "$name: $tensorName is ${tensor["dtype"]}, not FLOAT32" }
            tensorName to (tensor["shape"] as List<*>).map { (it as JsonNumber).toInt() }
          }
      val outputs = tensors(file["outputs"])
      require(outputs.size == 1) { "$name has ${outputs.size} outputs" }
      return Graph(
        name,
        signature,
        (file["bytes"] as JsonNumber).literal.toLong(),
        file["sha256"] as String,
        LinkedHashMap(tensors(file["inputs"]).toMap()),
        outputs.single(),
      )
    }
    tower = graph("vision_tower")
    projector = graph("projector")
    val patchRows = listOf(1, D1Vision.MAX_PATCHES, D1Vision.PATCH_VALUES)
    require(
      tower.inputs.toList() ==
        listOf("pixels" to patchRows, "pos" to listOf(1, D1Vision.MAX_PATCHES, D1Vision.HIDDEN),
          "mask" to listOf(1, D1Vision.MAX_PATCHES)) &&
        tower.output == ("features" to listOf(1, D1Vision.MAX_PATCHES, D1Vision.HIDDEN))
    ) {
      "contract.json vision_tower takes ${tower.inputs} -> ${tower.output}; this app lays out pixels / pos [1, 1024, 768], mask [1, 1024] -> features [1, 1024, 768]"
    }
    require(
      projector.inputs.toList() == listOf("soft" to listOf(1, D1Vision.PROJECTOR_ROWS, D1Vision.UNSHUFFLED)) &&
        projector.output == ("prefix" to listOf(1, D1Vision.PROJECTOR_ROWS, D1Rows.PREFIX_WIDTH))
    ) {
      "contract.json projector takes ${projector.inputs} -> ${projector.output}; this app lays out soft [1, 256, 3072] -> prefix [1, 256, 1024]"
    }
    val table = root["vision_position_table"] as Map<*, *>
    tableFile = table["file"] as String
    tableSha256 = table["sha256"] as String
    require(table["dtype"] == "float32") { "position table dtype ${table["dtype"]}" }
    require(
      (table["shape"] as List<*>).map { (it as JsonNumber).toInt() } ==
        listOf(D1Vision.TABLE_SIDE, D1Vision.TABLE_SIDE, D1Vision.HIDDEN)
    ) {
      "position table shape ${table["shape"]}"
    }
  }

  companion object {
    fun read(file: File): D1VisionContract = D1VisionContract(D1Json.parse(file.readBytes()))
  }
}

/** One crop of a picture's prefix run: its size, patch grid, cells, and its two graph calls. */
class D1CropRun(
  val index: Int,
  val height: Int,
  val width: Int,
  val gridHeight: Int,
  val gridWidth: Int,
  val cells: Int,
  val positionsCached: Boolean,
  val towerMs: Double,
  val projectorMs: Double,
  val towerCall: D1GraphCall?,
  val projectorCall: D1GraphCall?,
)

/**
 * A picture's prefix rows [P x 1024] (row-major) and the wall time of each host step and graph, in
 * nanoseconds: the layout and resize of the crops ([resizeNanos]), the patches of every crop
 * ([patchesNanos]), the position tables, the tower calls, the unshuffle and projector input, the
 * projector calls.
 */
class D1PrefixRun(
  val prefix: FloatArray,
  val rows: Int,
  val layout: D1Layout,
  val crops: List<D1CropRun>,
  val resizeNanos: Long,
  val positionsNanos: Long,
  val towerNanos: Long,
  val unshuffleNanos: Long,
  val projectorNanos: Long,
  val patchesNanos: Long = 0L,
) {
  /** The layout, the resize and the patches together (the debug runs' `resize_patches_ms`). */
  val resizePatchesNanos: Long
    get() = resizeNanos + patchesNanos

  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "P" to rows,
      "layout" to
        linkedMapOf(
          "grid_cols" to layout.gridCols,
          "grid_rows" to layout.gridRows,
          "thumb_h" to layout.thumbHeight,
          "thumb_w" to layout.thumbWidth,
          "tiled" to layout.tiled,
        ),
      "resize_patches_ms" to resizePatchesNanos / NANOS_PER_MS,
      "resize_ms" to resizeNanos / NANOS_PER_MS,
      "patches_ms" to patchesNanos / NANOS_PER_MS,
      "pos_ms" to positionsNanos / NANOS_PER_MS,
      "tower_ms" to towerNanos / NANOS_PER_MS,
      "unshuffle_ms" to unshuffleNanos / NANOS_PER_MS,
      "projector_ms" to projectorNanos / NANOS_PER_MS,
      "crops" to
        crops.map {
          linkedMapOf(
            "k" to it.index,
            "hw" to listOf(it.height, it.width),
            "grid" to listOf(it.gridHeight, it.gridWidth),
            "cells" to it.cells,
            "pos_cached" to it.positionsCached,
            "tower_ms" to it.towerMs,
            "tower_run_ms" to it.towerCall?.runMs,
            "projector_ms" to it.projectorMs,
            "projector_run_ms" to it.projectorCall?.runMs,
          )
        },
    )

  private companion object {
    const val NANOS_PER_MS = 1e6
  }
}

/**
 * `image_prefix` of `d1_vision_host.py`, Android-free (the graphs come in as functions, so the JVM
 * tests run it with the Python host's own graph outputs): the crops (tiles row-major, then the
 * thumbnail), and per crop the patches, the padded position rows, the tower, the unshuffle of its
 * real patches, the projector, its first (ph/2)(pw/2) rows; crops' rows concatenated.
 */
object D1VisionPrefix {
  /** One graph call's output and, on the device, the call itself (its times). */
  class Output(val values: FloatArray, val call: D1GraphCall?)

  /** What a caller checks per crop, outside the timed spans: the crop, its patches and positions. */
  fun interface Observer {
    fun crop(index: Int, crop: D1Rgb, patches: D1Patches, positions: FloatArray)
  }

  /**
   * [positions] gives a grid's padded position rows (and whether they came from a cache), [tower]
   * maps (patches, positions) to features [1024 x 768], [projector] maps soft [256 x 3072] to
   * [256 x 1024].
   */
  fun run(
    rgb: D1Rgb,
    positions: (Int, Int) -> Pair<FloatArray, Boolean>,
    tower: (D1Patches, FloatArray) -> Output,
    projector: (FloatArray) -> Output,
    observer: Observer? = null,
  ): D1PrefixRun {
    var resize = 0L
    var patching = 0L
    var position = 0L
    var towerTotal = 0L
    var unshuffle = 0L
    var projectorTotal = 0L
    var start = System.nanoTime()
    val (crops, layout) = D1Vision.crops(rgb)
    resize += System.nanoTime() - start
    val total = crops.sumOf { (it.height / D1Vision.PATCH / 2) * (it.width / D1Vision.PATCH / 2) }
    val prefix = FloatArray(total * D1Rows.PREFIX_WIDTH)
    val runs = ArrayList<D1CropRun>()
    var row = 0
    for ((index, crop) in crops.withIndex()) {
      start = System.nanoTime()
      val patches = D1Vision.toPatches(crop)
      patching += System.nanoTime() - start
      start = System.nanoTime()
      val (pos, cached) = positions(patches.gridHeight, patches.gridWidth)
      position += System.nanoTime() - start
      observer?.crop(index, crop, patches, pos)
      start = System.nanoTime()
      val features = tower(patches, pos)
      val towerNanos = System.nanoTime() - start
      towerTotal += towerNanos
      start = System.nanoTime()
      val cells = D1Vision.pixelUnshuffle(features.values, patches.gridHeight, patches.gridWidth)
      val soft = D1Vision.projectorInput(cells, patches.cells)
      unshuffle += System.nanoTime() - start
      start = System.nanoTime()
      val projected = projector(soft)
      val projectorNanos = System.nanoTime() - start
      projectorTotal += projectorNanos
      start = System.nanoTime()
      System.arraycopy(projected.values, 0, prefix, row * D1Rows.PREFIX_WIDTH, patches.cells * D1Rows.PREFIX_WIDTH)
      unshuffle += System.nanoTime() - start
      row += patches.cells
      runs.add(
        D1CropRun(
          index,
          crop.height,
          crop.width,
          patches.gridHeight,
          patches.gridWidth,
          patches.cells,
          cached,
          towerNanos / 1e6,
          projectorNanos / 1e6,
          features.call,
          projected.call,
        )
      )
    }
    check(row == total) { "prefix rows $row, expected $total" }
    return D1PrefixRun(prefix, total, layout, runs, resize, position, towerTotal, unshuffle, projectorTotal, patching)
  }
}

/**
 * One graph compile of the picture path: which graph, where it was asked for and where it runs, its
 * time, the GPU's error after a fallback, the memory right before and after it ([D1Device.memory]).
 */
class D1VisionCompile(
  val graph: String,
  val requested: D1Backend,
  val backend: D1Backend,
  val precision: D1Precision,
  val compileMs: Double,
  val gpuFailure: String?,
  val memoryBefore: Map<String, Any?>,
  val memoryAfter: Map<String, Any?>,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "graph" to graph,
      "requested" to requested.wireName,
      "backend" to backend.wireName,
      "precision" to if (backend == D1Backend.GPU) precision.wireName else null,
      "compile_ms" to compileMs,
      "gpu_failure" to gpuFailure,
      "memory_before" to memoryBefore,
      "memory_after" to memoryAfter,
    )
}

/**
 * The picture path on the phone: the position table read from `files/` (sha256 = contract.json),
 * the vision tower and the projector compiled once on [backend] at [precision] and kept, and
 * [imagePrefix] for each picture. A graph the GPU cannot compile or run runs on the CPU (four
 * threads) instead (`GPU_FALLBACK` in logcat). Use only on [D1Runtime.dispatcher].
 */
class D1VisionEngine
private constructor(
  private val context: Context,
  val contract: D1VisionContract,
  val table: FloatArray,
  /** Wall time of reading and checking the position table, in milliseconds. */
  val tableMs: Double,
  val backend: D1Backend,
  val precision: D1Precision,
) : Closeable {
  /** Every compile of this engine, in order. */
  val compiles = ArrayList<D1VisionCompile>()

  private var tower: D1Graph = compile(contract.tower, "vision_tower", backend, null)
  private var projector: D1Graph = compile(contract.projector, "projector", backend, null)
  private val positionsCache = LinkedHashMap<Int, FloatArray>()

  /** Where the tower and the projector run now. */
  val towerBackend: D1Backend
    get() = tower.backend

  val projectorBackend: D1Backend
    get() = projector.backend

  /** The picture's prefix rows [P x 1024] and the time of each step ([D1VisionPrefix.run]). */
  fun imagePrefix(rgb: D1Rgb, observer: D1VisionPrefix.Observer? = null): D1PrefixRun =
    D1VisionPrefix.run(
      rgb,
      positions = ::positions,
      tower = { patches, pos ->
        val call = runTower(mapOf("pixels" to patches.pixels, "pos" to pos, "mask" to patches.mask))
        D1VisionPrefix.Output(call.values, call)
      },
      projector = { soft ->
        val call = runProjector(mapOf("soft" to soft))
        D1VisionPrefix.Output(call.values, call)
      },
      observer = observer,
    )

  /** A grid's padded position rows, kept for the last [POSITIONS_CACHED] grids. */
  private fun positions(gridHeight: Int, gridWidth: Int): Pair<FloatArray, Boolean> {
    val key = gridHeight * KEY_STRIDE + gridWidth
    positionsCache.remove(key)?.let {
      positionsCache[key] = it
      return it to true
    }
    val rows = D1Vision.positionsPadded(table, gridHeight, gridWidth)
    positionsCache[key] = rows
    while (positionsCache.size > POSITIONS_CACHED) positionsCache.remove(positionsCache.keys.first())
    return rows to false
  }

  private fun runTower(feeds: Map<String, FloatArray>): D1GraphCall =
    try {
      tower.run(feeds)
    } catch (failure: Exception) {
      if (tower.backend != D1Backend.GPU) throw failure
      val reason = "run vision_tower: ${D1Decider.describe(failure)}"
      D1Demo.gpuFallback(reason)
      tower.close()
      tower = compile(contract.tower, "vision_tower", D1Backend.CPU, reason)
      tower.run(feeds)
    }

  private fun runProjector(feeds: Map<String, FloatArray>): D1GraphCall =
    try {
      projector.run(feeds)
    } catch (failure: Exception) {
      if (projector.backend != D1Backend.GPU) throw failure
      val reason = "run projector: ${D1Decider.describe(failure)}"
      D1Demo.gpuFallback(reason)
      projector.close()
      projector = compile(contract.projector, "projector", D1Backend.CPU, reason)
      projector.run(feeds)
    }

  private fun compile(
    graph: D1VisionContract.Graph,
    name: String,
    target: D1Backend,
    earlierFailure: String?,
  ): D1Graph {
    val before = D1Device.memory(context)
    val compiled =
      D1Graph.create(
        context,
        File(context.filesDir, graph.file),
        graph.signature,
        graph.inputs,
        graph.output,
        target,
        precision,
        fallback = target == D1Backend.GPU,
      )
    if (compiled.gpuFailure != null) D1Demo.gpuFallback("compile $name: ${compiled.gpuFailure}")
    compiles.add(
      D1VisionCompile(
        name,
        target,
        compiled.backend,
        precision,
        compiled.compileMs,
        compiled.gpuFailure ?: earlierFailure,
        before,
        D1Device.memory(context),
      )
    )
    return compiled
  }

  override fun close() {
    try {
      tower.close()
    } finally {
      projector.close()
    }
  }

  companion object {
    private const val POSITIONS_CACHED = 4
    private const val KEY_STRIDE = 4096

    /**
     * Reads the picture path's contract entries and the position table from `files/` (the table's
     * sha256 checked against contract.json), then compiles the tower and the projector.
     */
    fun open(context: Context, backend: D1Backend, precision: D1Precision): D1VisionEngine {
      val files = context.filesDir
      val contract = D1VisionContract.read(File(files, D1Contract.FILE))
      val start = System.nanoTime()
      val table = D1Npy.positionTable(File(files, contract.tableFile), contract.tableSha256)
      val tableMs = (System.nanoTime() - start) / 1e6
      return D1VisionEngine(context.applicationContext, contract, table, tableMs, backend, precision)
    }
  }
}
