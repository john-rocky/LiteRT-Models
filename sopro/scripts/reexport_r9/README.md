# Exact layout variants
Use the same pinned Python environment and source checkpoint as the canonical conversion scripts.
Style attention folds its constant-query reshape before optional FLOAT_CASTING.
The fp32 style file is the application candidate; its wfp16 twin is retained for comparison.
Streaming start, step and flush promote rank-three tensor paths to rank four.
Streaming promotion updates only layout metadata, axes and integer shape constants.
Original floating buffers and operation order remain unchanged.
Supply each published fp32 streaming file with `--source-graph`; its checksum is verified.
Export fp32 before `--precision wfp16`; no checkpoint download or device command runs implicitly.
Run `python -B scripts/reexport_r9/export.py --help` from the module root for options.
