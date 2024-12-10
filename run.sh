#!/bin/bash
PARAMETERS=(0.15 0.20 0.25 0.30)
GPUS=(0 1 2 3)
for i in "${!PARAMETERS[@]}"; do
    # 设置CUDA_VISIBLE_DEVICES环境变量
    export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
    python FNorm4MSIFineTune.py --visible_ratio="${PARAMETERS[$i]}" &
done
# 等待所有后台任务完成
wait