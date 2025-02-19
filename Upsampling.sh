# Create an array to hold process ids (PIDs)
PIDS=()

# Capture the SIGINT signal and define the handler
trap 'handle_interrupt' INT

handle_interrupt() {
    echo "Interrupt signal received. Terminating all subprocesses..."
    for pid in "${PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    exit 1
}

NAMES=('Table' 'Airplane' 'Chair' 'Lamp')
GPUS=(0 1 2 3)
# Loop through the cases and gpus arrays
for i in "${!NAMES[@]}"; do
    export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
    python FNorm4Bunny --name="${NAMES[$i]}" &
    # ablation
    # python ablation/Upsampling.py --name="${NAMES[$i]}" &
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