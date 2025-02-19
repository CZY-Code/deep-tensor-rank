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

CASES=(1 2 3 4 5)
GPUS=(0 1 2 3 3)
# 循环遍历 cases 和 gpus 数组
for i in "${!CASES[@]}"; do
    export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
    # 启动 Python 脚本并将它放到后台运行
    # python FNorm4DCmallDenoising.py --save --case="${CASES[$i]}" &
    # python FNorm4PaviaUDenoising.py --case="${CASES[$i]}" &
    # python FNorm4CAVEDenoising.py --save --case="${CASES[$i]}" &
    # 复现
    # python Demos/LRTFR/LRTFR_Denoising.py --save --case="${CASES[$i]}" &
    # ablation
    python ablation/Denoising.py --case="${CASES[$i]}" &
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