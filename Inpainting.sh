#!/bin/bash
PARAMETERS=(0.10 0.15 0.20 0.25 0.30)
GPUS=(0 1 2 3 0)
# PARAMETERS=(0.30)
# GPUS=(0)
# Create an array to hold process ids (PIDs)
PIDS=()

# 捕获 SIGINT 信号并定义处理函数
trap 'handle_interrupt' INT

handle_interrupt() {
    echo "Interrupt signal received. Terminating all subprocesses..."
    for pid in "${PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    exit 1
}

# Loop through the cases and gpus arrays
for i in "${!PARAMETERS[@]}"; do
    export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
    # python FNorm4RGB.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # python FNorm4MSI.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # python FNorm4Video.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # Reproduct SOTA method
    # python Demos/LRTFR/LRTFR_Inpainting.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # ablation
    python ablation/Inpainting.py --visible_ratio="${PARAMETERS[$i]}" &
    # Save process ID
    PIDS+=($!)
done

# Wait for all background tasks to complete and check for any failed tasks
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || let "FAIL+=1"
done

# Check for failed tasks
if [ "$FAIL" -gt 0 ]; then
    echo "Some tasks failed."
    exit 1
else
    echo "All tasks completed successfully."
fi