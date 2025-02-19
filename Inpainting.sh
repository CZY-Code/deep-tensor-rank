#!/bin/bash
PARAMETERS=(0.10 0.15 0.20 0.25 0.30)
GPUS=(0 1 2 3 0)
# PARAMETERS=(0.30)
# GPUS=(0)
# 创建一个数组来保存进程ID (PIDs)
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

# 循环遍历 cases 和 gpus 数组
for i in "${!PARAMETERS[@]}"; do
    export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
    # python FNorm4RGB.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # python FNorm4MSI.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # python FNorm4Video.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # Reproduct SOTA method
    # python Demos/LRTFR/LRTFR_Inpainting.py --save --visible_ratio="${PARAMETERS[$i]}" &
    # ablation
    python ablation/Inpainting.py --visible_ratio="${PARAMETERS[$i]}" &
    # 保存进程ID
    PIDS+=($!)
done

# 等待所有后台任务完成，并检查是否有失败的任务
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || let "FAIL+=1"
done

# 检查是否有失败的任务
if [ "$FAIL" -gt 0 ]; then
    echo "Some tasks failed."
    exit 1
else
    echo "All tasks completed successfully."
fi