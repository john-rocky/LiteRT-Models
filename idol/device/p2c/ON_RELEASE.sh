#!/bin/bash
# ON_RELEASE.sh: written by npu_chain.sh pid 49557 at 2026-11-09 20:26:19 for idol-p2c-s26 on RFGL80R6A6H. keeper.sh runs it only when
# it releases the hold at its limit (no RELEASE_HOLD from the chain), before the release: this chain's own
# processes, app and stage, read before and after.
A() { adb -s RFGL80R6A6H "$@" < /dev/null; }
S=t=\$\(dumpsys\ activity\ activities\ 2\>\ /dev/null\ \|\ grep\ -m1\ -E\ \'topResumedActivity\|ResumedActivity:\'\)\;\ \[\ -n\ \"\$t\"\ \]\ \|\|\ t=\"top\ task\ \(none\ resumed\):\ \$\(dumpsys\ activity\ activities\ 2\>\ /dev/null\ \|\ grep\ -m1\ -E\ \'Hist\ +#0:\ ActivityRecord\'\)\"\;\ echo\ \"front=\$t\"\;\ echo\ \"mem\ MemAvailable=\$\(grep\ MemAvailable\ /proc/meminfo\ \|\ tr\ -dc\ 0-9\)\ kgsl_page_alloc=\$\(cat\ /sys/class/kgsl/kgsl/page_alloc\ 2\>\ /dev/null\)\ mali_total_gpu_mem=\$\(cat\ /sys/class/misc/mali0/device/total_gpu_mem\ 2\>\ /dev/null\)\ mali_dma_buf_gpu_mem=\$\(cat\ /sys/class/misc/mali0/device/dma_buf_gpu_mem\ 2\>\ /dev/null\)\"
echo "before $(date "+%F %T"): $(A shell "$S" | tr -d "\r" | tr "\n" " ")"
A shell for\ p\ in\ \$\(pidof\ r3_rt_runner\ 2\>\ /dev/null\)\;\ do\ kill\ -9\ \$p\ \&\&\ echo\ \"runner\ pid\ \$p\ -\>\ kill\ -9\"\;\ done\;\ rm\ -rf\ /data/local/tmp/idol_p2c\;\ echo\ \"stage\ left:\ \$\(ls\ -d\ /data/local/tmp/idol_p2c\ 2\>\&1\)\;\ runner\ left:\ \'\$\(pidof\ r3_rt_runner\ 2\>\ /dev/null\)\'\"
echo "after $(date "+%F %T"): $(A shell "$S" | tr -d "\r" | tr "\n" " ")"
