| row | runtime and options | wrapper change needed | det | pose | seg |
|---|---|---|---|---|---|
| A1d | 1.4.0 Interpreter, CPU, threads unset | none (3.0.1 makes this call) | 24.87 | 7.82 | does not load |
| A1t4 | 1.4.0 Interpreter, CPU, 4 threads | set a thread count (3.0.1 does not) | 27.11 | 8.86 | does not load |
| A1x | 1.4.0 Interpreter + XNNPACK delegate, the delegate's default threads | add the XNNPACK delegate (3.0.1 does not) | 10.31 | 2.32 | 1.61 |
| A1xt4 | 1.4.0 Interpreter + XNNPACK delegate, 4 threads | add the XNNPACK delegate and a thread count (3.0.1 does neither) | 9.01 | 1.92 | 1.18 |
| A2d | 1.4.0 GPU delegate, default options (max precision first) | none (3.0.1 makes this call) | 19.54 | 7.08 | does not load |
| A2h | 1.4.0 GPU delegate, precision loss allowed (min latency first) | set the GPU options above (3.0.1 does not) | 13.71 FAIL | 6.21 | does not load |
| A3n | 1.4.0 NNAPI delegate | none (3.0.1 makes this call) | 24.58 | 7.78 | does not load |
| B1d | 2.3.2 CompiledModel, CPU, default | rewrite to the CompiledModel API | 8.68 | 2.23 | 1.69 |
| B1t4 | 2.3.2 CompiledModel, CPU, 4 threads | rewrite to the CompiledModel API | 8.93 | 1.89 | 1.42 |
| B2d | 2.3.2 CompiledModel, GPU + CPU, default precision | rewrite to the CompiledModel API | 13.20 FAIL | 2.63 FAIL | 1.00 |
| B2h | 2.3.2 CompiledModel, GPU + CPU, fp16 | rewrite to the CompiledModel API | 13.27 FAIL | 2.65 FAIL | 1.01 |
| B2f | 2.3.2 CompiledModel, GPU + CPU, fp32 | rewrite to the CompiledModel API | 18.98 FAIL | 3.27 FAIL | 1.23 |
| B2g | 2.3.2 CompiledModel, GPU alone, default precision | rewrite to the CompiledModel API | does not load | does not load | 0.99 |
| B2dH | 2.3.2 CompiledModel, GPU + CPU, default precision, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | 13.36 FAIL | 2.65 FAIL | 1.08 |
| B2hH | 2.3.2 CompiledModel, GPU + CPU, fp16, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | 13.25 FAIL | 2.65 FAIL | 1.11 |
| B2fH | 2.3.2 CompiledModel, GPU + CPU, fp32, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | 18.77 FAIL | 3.32 FAIL | 1.34 |
| B3c | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, no cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | 17.37 | 16.59 FAIL | failed |
| B3 | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone, read from B3c's cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | 18.05 | 17.46 FAIL | - |
| B3b | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, HTP burst mode | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app; the HTP burst mode in the Qualcomm options | 4.15 | 2.38 FAIL | - |
| X1t4 | 2.3.2 libLiteRt.so through the TFLite C API, CPU, 4 threads | version bump, the new .so name in CMake, headers kept from 1.4.0, a thread count; no GPU or NNAPI delegate (their C functions are not in the 2.3.2 libraries) | 10.88 | 3.37 | does not load |

- det: 1.4.0 A1xt4 9.007 ms (p10-p90 7.58-9.73); 2.3.2 B3b 4.146 ms (p10-p90 4.00-4.28); time 1.4.0 ÷ 2.3.2 = 2.17.
- pose: 1.4.0 A1xt4 1.919 ms (p10-p90 1.80-2.01); 2.3.2 B1t4 1.893 ms (p10-p90 1.74-1.99); time 1.4.0 ÷ 2.3.2 = 1.01.
- seg: 1.4.0 A1xt4 1.182 ms (p10-p90 1.06-1.22); 2.3.2 B2g 0.992 ms (p10-p90 0.67-1.02); time 1.4.0 ÷ 2.3.2 = 1.19.

| model | row | runtime and options | wrapper change needed | status | median ms (p10-p90) | load ms | first call ms | CPU threads used | agreement |
|---|---|---|---|---|---|---|---|---|---|
| det | A1d | 1.4.0 Interpreter, CPU, threads unset | none (3.0.1 makes this call) | ran | 24.87 (24.35-25.87) | 5.8 | 62.53 | 1 | PASS |
| det | A1t4 | 1.4.0 Interpreter, CPU, 4 threads | set a thread count (3.0.1 does not) | ran | 27.11 (23.59-34.34) | 6.6 | 34.84 | 4 | PASS |
| det | A1x | 1.4.0 Interpreter + XNNPACK delegate, the delegate's default threads | add the XNNPACK delegate (3.0.1 does not) | ran | 10.31 (10.12-10.52) | 16.5 | 26.36 | 1 | PASS |
| det | A1xt4 | 1.4.0 Interpreter + XNNPACK delegate, 4 threads | add the XNNPACK delegate and a thread count (3.0.1 does neither) | ran | 9.01 (7.58-9.73) | 18.0 | 19.55 | 4 | PASS |
| det | A2d | 1.4.0 GPU delegate, default options (max precision first) | none (3.0.1 makes this call) | ran | 19.54 (16.28-21.76) | 529.0 | 32.54 | 2 | PASS |
| det | A2h | 1.4.0 GPU delegate, precision loss allowed (min latency first) | set the GPU options above (3.0.1 does not) | ran | 13.71 (12.73-14.78) | 512.8 | 33.63 | 2 | FAIL |
| det | A3n | 1.4.0 NNAPI delegate | none (3.0.1 makes this call) | ran | 24.58 (24.11-24.85) | 87.5 | 29.57 | 1 | PASS |
| det | B1d | 2.3.2 CompiledModel, CPU, default | rewrite to the CompiledModel API | ran | 8.68 (8.42-8.96) | 172.5 | 11.67 | 1 | PASS |
| det | B1t4 | 2.3.2 CompiledModel, CPU, 4 threads | rewrite to the CompiledModel API | ran | 8.93 (7.90-9.36) | 76.1 | 10.72 | 4 | PASS |
| det | B2d | 2.3.2 CompiledModel, GPU + CPU, default precision | rewrite to the CompiledModel API | ran | 13.20 (11.10-14.46) | 734.4 | 14.39 | 2 | FAIL |
| det | B2h | 2.3.2 CompiledModel, GPU + CPU, fp16 | rewrite to the CompiledModel API | ran | 13.27 (11.13-15.08) | 736.9 | 15.06 | 2 | FAIL |
| det | B2f | 2.3.2 CompiledModel, GPU + CPU, fp32 | rewrite to the CompiledModel API | ran | 18.98 (13.59-21.52) | 631.9 | 16.12 | 2 | FAIL |
| det | B2g | 2.3.2 CompiledModel, GPU alone, default precision | rewrite to the CompiledModel API | does not load: LiteRtCreateCompiledModel status 504 | - | - | - | - | - |
| det | B2dH | 2.3.2 CompiledModel, GPU + CPU, default precision, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 13.36 (11.42-14.85) | 709.1 | 14.79 | 2 | FAIL |
| det | B2hH | 2.3.2 CompiledModel, GPU + CPU, fp16, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 13.25 (11.58-15.34) | 702.2 | 14.44 | 2 | FAIL |
| det | B2fH | 2.3.2 CompiledModel, GPU + CPU, fp32, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 18.77 (14.87-21.27) | 656.5 | 16.49 | 2 | FAIL |
| det | B3c | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, no cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | ran | 17.37 (13.29-20.60) | 729.2 | 23.40 | 1 | PASS |
| det | B3 | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone, read from B3c's cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | ran | 18.05 (14.11-20.58) | 170.1 | 27.28 | 1 | PASS |
| det | B3b | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, HTP burst mode | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app; the HTP burst mode in the Qualcomm options | ran | 4.15 (4.00-4.28) | 733.6 | 7.15 | 1 | PASS |
| det | X1t4 | 2.3.2 libLiteRt.so through the TFLite C API, CPU, 4 threads | version bump, the new .so name in CMake, headers kept from 1.4.0, a thread count; no GPU or NNAPI delegate (their C functions are not in the 2.3.2 libraries) | ran | 10.88 (10.32-11.56) | 14.0 | 27.86 | 4 | PASS |
| pose | A1d | 1.4.0 Interpreter, CPU, threads unset | none (3.0.1 makes this call) | ran | 7.82 (7.79-8.00) | 4.5 | 24.84 | 1 | PASS |
| pose | A1t4 | 1.4.0 Interpreter, CPU, 4 threads | set a thread count (3.0.1 does not) | ran | 8.86 (7.88-12.19) | 3.7 | 11.06 | 4 | PASS |
| pose | A1x | 1.4.0 Interpreter + XNNPACK delegate, the delegate's default threads | add the XNNPACK delegate (3.0.1 does not) | ran | 2.32 (2.30-2.44) | 10.7 | 7.30 | 1 | PASS |
| pose | A1xt4 | 1.4.0 Interpreter + XNNPACK delegate, 4 threads | add the XNNPACK delegate and a thread count (3.0.1 does neither) | ran | 1.92 (1.80-2.01) | 12.5 | 4.82 | 4 | PASS |
| pose | A2d | 1.4.0 GPU delegate, default options (max precision first) | none (3.0.1 makes this call) | ran | 7.08 (5.84-7.49) | 659.2 | 11.10 | 2 | PASS |
| pose | A2h | 1.4.0 GPU delegate, precision loss allowed (min latency first) | set the GPU options above (3.0.1 does not) | ran | 6.21 (5.94-6.52) | 677.5 | 10.50 | 2 | PASS |
| pose | A3n | 1.4.0 NNAPI delegate | none (3.0.1 makes this call) | ran | 7.78 (7.62-7.94) | 78.1 | 10.03 | 1 | PASS |
| pose | B1d | 2.3.2 CompiledModel, CPU, default | rewrite to the CompiledModel API | ran | 2.23 (2.22-2.25) | 288.0 | 7.15 | 1 | PASS |
| pose | B1t4 | 2.3.2 CompiledModel, CPU, 4 threads | rewrite to the CompiledModel API | ran | 1.89 (1.74-1.99) | 65.2 | 3.38 | 4 | PASS |
| pose | B2d | 2.3.2 CompiledModel, GPU + CPU, default precision | rewrite to the CompiledModel API | ran | 2.63 (2.10-2.74) | 730.1 | 8.00 | 2 | FAIL |
| pose | B2h | 2.3.2 CompiledModel, GPU + CPU, fp16 | rewrite to the CompiledModel API | ran | 2.65 (2.59-2.77) | 707.1 | 7.29 | 2 | FAIL |
| pose | B2f | 2.3.2 CompiledModel, GPU + CPU, fp32 | rewrite to the CompiledModel API | ran | 3.27 (3.01-3.42) | 536.3 | 8.13 | 2 | FAIL |
| pose | B2g | 2.3.2 CompiledModel, GPU alone, default precision | rewrite to the CompiledModel API | does not load: LiteRtCreateCompiledModel status 504 | - | - | - | - | - |
| pose | B2dH | 2.3.2 CompiledModel, GPU + CPU, default precision, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 2.65 (2.60-2.74) | 697.2 | 7.22 | 2 | FAIL |
| pose | B2hH | 2.3.2 CompiledModel, GPU + CPU, fp16, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 2.65 (2.09-2.75) | 701.9 | 7.34 | 2 | FAIL |
| pose | B2fH | 2.3.2 CompiledModel, GPU + CPU, fp32, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 3.32 (3.18-3.62) | 528.5 | 8.16 | 2 | FAIL |
| pose | B3c | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, no cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | ran | 16.59 (14.94-21.34) | 644.8 | 65.50 | 1 | FAIL |
| pose | B3 | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone, read from B3c's cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | ran | 17.46 (15.74-22.10) | 196.0 | 47.36 | 1 | FAIL |
| pose | B3b | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, HTP burst mode | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app; the HTP burst mode in the Qualcomm options | ran | 2.38 (2.36-2.43) | 494.5 | 8.56 | 1 | FAIL |
| pose | X1t4 | 2.3.2 libLiteRt.so through the TFLite C API, CPU, 4 threads | version bump, the new .so name in CMake, headers kept from 1.4.0, a thread count; no GPU or NNAPI delegate (their C functions are not in the 2.3.2 libraries) | ran | 3.37 (3.08-3.49) | 11.2 | 10.65 | 4 | PASS |
| seg | A1d | 1.4.0 Interpreter, CPU, threads unset | none (3.0.1 makes this call) | does not load: TfLiteInterpreterAllocateTensors status 7 | - | - | - | - | - |
| seg | A1t4 | 1.4.0 Interpreter, CPU, 4 threads | set a thread count (3.0.1 does not) | does not load: TfLiteInterpreterAllocateTensors status 7 | - | - | - | - | - |
| seg | A1x | 1.4.0 Interpreter + XNNPACK delegate, the delegate's default threads | add the XNNPACK delegate (3.0.1 does not) | ran | 1.61 (1.60-1.71) | 6.8 | 5.97 | 1 | PASS (Mac CPU reference) |
| seg | A1xt4 | 1.4.0 Interpreter + XNNPACK delegate, 4 threads | add the XNNPACK delegate and a thread count (3.0.1 does neither) | ran | 1.18 (1.06-1.22) | 7.8 | 9.16 | 4 | PASS (Mac CPU reference) |
| seg | A2d | 1.4.0 GPU delegate, default options (max precision first) | none (3.0.1 makes this call) | does not load: TfLiteInterpreterCreate returned null | - | - | - | - | - |
| seg | A2h | 1.4.0 GPU delegate, precision loss allowed (min latency first) | set the GPU options above (3.0.1 does not) | does not load: TfLiteInterpreterCreate returned null | - | - | - | - | - |
| seg | A3n | 1.4.0 NNAPI delegate | none (3.0.1 makes this call) | does not load: TfLiteInterpreterCreate returned null | - | - | - | - | - |
| seg | B1d | 2.3.2 CompiledModel, CPU, default | rewrite to the CompiledModel API | ran | 1.69 (1.58-1.70) | 216.0 | 3.55 | 1 | PASS (Mac CPU reference) |
| seg | B1t4 | 2.3.2 CompiledModel, CPU, 4 threads | rewrite to the CompiledModel API | ran | 1.42 (1.38-1.46) | 77.2 | 5.03 | 4 | PASS (Mac CPU reference) |
| seg | B2d | 2.3.2 CompiledModel, GPU + CPU, default precision | rewrite to the CompiledModel API | ran | 1.00 (0.96-1.04) | 620.0 | 1.82 | 2 | PASS (Mac CPU reference) |
| seg | B2h | 2.3.2 CompiledModel, GPU + CPU, fp16 | rewrite to the CompiledModel API | ran | 1.01 (0.99-1.04) | 534.3 | 6.13 | 2 | PASS (Mac CPU reference) |
| seg | B2f | 2.3.2 CompiledModel, GPU + CPU, fp32 | rewrite to the CompiledModel API | ran | 1.23 (0.95-1.26) | 393.6 | 6.00 | 2 | PASS (Mac CPU reference) |
| seg | B2g | 2.3.2 CompiledModel, GPU alone, default precision | rewrite to the CompiledModel API | ran | 0.99 (0.67-1.02) | 475.4 | 6.45 | 2 | PASS (Mac CPU reference) |
| seg | B2dH | 2.3.2 CompiledModel, GPU + CPU, default precision, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 1.08 (0.68-1.14) | 474.2 | 6.58 | 2 | PASS (Mac CPU reference) |
| seg | B2hH | 2.3.2 CompiledModel, GPU + CPU, fp16, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 1.11 (1.01-1.15) | 509.2 | 6.20 | 2 | PASS (Mac CPU reference) |
| seg | B2fH | 2.3.2 CompiledModel, GPU + CPU, fp32, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | ran | 1.34 (1.22-1.39) | 404.9 | 6.18 | 2 | PASS (Mac CPU reference) |
| seg | B3c | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, no cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | failed: run: LiteRtRunCompiledModel status 3 | - | 632.5 | - | - | - |
| seg | X1t4 | 2.3.2 libLiteRt.so through the TFLite C API, CPU, 4 threads | version bump, the new .so name in CMake, headers kept from 1.4.0, a thread count; no GPU or NNAPI delegate (their C functions are not in the 2.3.2 libraries) | does not load: TfLiteInterpreterAllocateTensors status 7 | - | - | - | - | - |
