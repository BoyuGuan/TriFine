# 检查是否提供了CUDA设备
if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_devices> <path_to_python_file> [additional_args]"
    exit 1
fi

# 获取CUDA设备并导出
CUDA_VISIBLE_DEVICES=$1
export CUDA_VISIBLE_DEVICES

# 移动到下一个参数
shift

# 检查是否提供了Python文件
if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_devices> <path_to_python_file> [additional_args]"
    exit 1
fi

PYTHON_FILE=$1
shift

# 计算CUDA_VISIBLE_DEVICES中的设备数量
NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
NUM_PROCESSES=$((NUM_PROCESSES + 1))

RANDOM_PORT=$(shuf -i 12000-13000 -n 1)

# 使用accelerate启动
accelerate launch \
    --main_process_port $RANDOM_PORT \
    --num_processes $NUM_PROCESSES \
    --num_machines 1 $PYTHON_FILE "$@"
