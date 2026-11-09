# Image to 3D on phone GPUs: one graph for a stack of blocks, and timings you can compare

[Idol](https://huggingface.co/speridlabs/idol) (speridlabs) turns one photo of an object into a 3D Gaussian splat.
Its LiteRT port runs the whole pipeline on the phone with `CompiledModel`, on a Galaxy S26 and a Pixel 11 Pro: the
sample app is [idol/](../idol/) and the graphs are on
[litert-community/Idol](https://huggingface.co/litert-community/Idol). Two things from that port may help with other
generative pipelines made of many blocks of the same shape, such as diffusion denoisers and decoder stacks:

1. Run a stack of same-shape blocks as one graph that takes each block's weights as inputs. The phone then compiles
   that graph once per photo. With a graph per block, the app closed compiled graphs to keep memory free and compiled
   the denoiser's blocks 96 times in one photo on the Galaxy S26.
2. Time whole runs only from the same starting temperature. On one phone, the same graphs made the same splat in
   267 s from a cool start (thermal status 0) and in 364 s from a warm one (thermal status 1).

## One graph for a stack of same-shape blocks

Idol's denoiser has 22 middle blocks of one shape (blocks 1–22 of 24), and its Gaussian decoder has six middle graphs
of one shape. litert-community/Idol holds two forms of the graphs. The first release's form, in `262k/`, has a graph
per block. The sample app keeps 2.5 GB of memory free and closes compiled graphs to make room, so those graphs did not
stay compiled: in one photo on the Galaxy S26 the app compiled the denoiser's joint blocks 96 times (37.2 s in all,
0.39 s each), and each compile took about 650 MB of the free memory.

The default form, at the repo root, runs each stack as one graph:

1. In PyTorch, write one module with the block's arithmetic that takes the block's weight tensors as extra inputs
   (`DenJointShared` in [`idol/scripts/idol_graphs.py`](../idol/scripts/idol_graphs.py)).
2. Check it against every block's own module on the same input before exporting.
   [`export_shared_joint.py`](../idol/scripts/export_shared_joint.py) stops unless every block's relative L2 error is
   below 1e-5 (3.9e-7 for the worst of the 22 blocks in its first run).
3. Export it once with litert-torch, with block 1's weights as the example inputs, and write all blocks' weights one
   after another into one little-endian float32 file, each block's tensors in argument order (`args_3` to `args_32`
   here; `args_0` to `args_2` are the block's activations) (`export_shared_joint.py`, excerpt):

   ```python
   shared = IG.DenJointShared(mids[0]).eval()
   # ... the per-block check and the weights file
   litert_torch.convert(shared, (h, t_mod, gate, *IG.shared_block_weights(mids[0]))).export(path)
   ```

4. In the app, compile the graph once, map the weights file, and before each call copy that block's weights into the
   graph's input buffers ([`BlockWeights.kt`](../idol/app/src/main/java/com/idol/BlockWeights.kt), excerpt):

   ```kotlin
   floats.position((base + offset).toInt())
   floats.get(a, 0, size)
   g.write(index, a, "w${block}_$k")
   ```

Find each weight input by its name, not by its position. The exported file's signature stores the inputs in string
order of their names (`args_0`, `args_1`, `args_10`, …, `args_2`), while LiteRT 2.3.0's `CompiledModel` lists the same
inputs in numeric order (`args_0`, `args_1`, `args_2`, …).

What it changed, measured with LiteRT 2.3.0:

- One block call on the Galaxy S26, both at GPU precision FP16 with fp32 accumulation: 899.8 ms with the shared graph
  against 901.7 ms with the block's own graph, with a bit-identical output; compiling the shared graph took 751 ms
  against 854 ms.
- With the denoiser's middle blocks in one graph, a photo compiles the denoiser's three joint graphs once each (7.6 s
  on the Pixel 11 Pro), where the per-block form had compiled joint blocks 96 times on the Galaxy S26.
- Sharing the Gaussian decoder's six middle graphs as well took the Pixel 11 Pro's first photo after start from
  406.8 s to 333.4 s: 15 graphs compiled during the photo (71.2 s) instead of 20 (126.5 s), and the Gaussian decoder
  stage took 24.1 s instead of 72.3 s.
- Files: the 22 per-block graphs of 56.1 MB each become one graph of 0.1 MB and one weights file of 1.23 GB.

## Timings you can compare

The same app with the same graphs (the first release's form) made a byte-identical splat of the same photo on the
same Galaxy S26 in 266.7 s and in 363.6 s. The 363.6 s photo started at thermal status 1 (skin above 40 °C) after an
earlier run; the 266.7 s photo started at thermal status 0 (skin 34.6 °C). Most of the 97 s was in the denoiser, which
runs on the GPU (277.7 s against 193.9 s). Both photos compiled about the same number of graphs in the same time; the
warm photo's app had also compiled two of the graphs before its photo started. On the same Galaxy S26, already hot
from back-to-back runs (thermal status 3), the default form took 329–351 s per photo against 158 s from a cool start.

The two photos the model card of litert-community/Idol compares for its two forms both started at thermal status 0
with the GPU clock cap at 1,300 MHz. To take such a photo:

- Read the GPU clock cap, the thermal status and the skin temperature every 10 s. On the Galaxy S26 (Adreno):

  ```bash
  adb shell 'echo CLK $(cat /sys/class/kgsl/kgsl-3d0/max_gpuclk) $(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel); dumpsys thermalservice | sed -n "/^Thermal Status/p; /Current temperatures from HAL/,/Current cooling devices/p"'
  ```

  Take the skin temperature from the `SKIN` line under "Current temperatures from HAL"; the "Cached temperatures"
  section can be old. Other GPUs expose their clock in other files.
- Start the photo after three reads in a row with the GPU clock cap at its top (1,300 MHz on the S26), thermal status
  0 and skin at or below 34.0 °C.
- Expect a long wait after a hot run: on the S26, 49 minutes passed between skin 44.5 °C (thermal status 3) at the end
  of one run and the first read of 34.0 °C, with the app stopped in between.
- Write the start state next to each number. To compare two forms of one graph, time them alternately in one sitting
  (A B B A) rather than in separate runs.

The card gives each form's time with its start state, and [idol/SPEED_LADDER.md](../idol/SPEED_LADDER.md) lists every
form that was measured, including the ones that did not ship.
