#!/bin/bash
# MPIIGaze 训练和 fine-tuning 便捷脚本

set -e

# 环境设置
export PYTHONPATH="${PYTHONPATH}:/home/xuhy/PycharmProjects/FAGE"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  FAGE MPIIGaze Training Script${NC}"
echo -e "${BLUE}========================================${NC}"

# 显示使用方法
usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  test              - 测试数据集加载"
    echo "  train             - Phase 1 训练 (GazeControlNet)"
    echo "  finetune          - 为单个 subject fine-tune adapter"
    echo "  finetune-all      - 为所有 subjects fine-tune adapters (p00-p14)"
    echo "  inference         - 推理对比 (baseline vs +adapter, 所有 subjects)"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 train"
    echo "  $0 finetune p12 /path/to/checkpoint.pth 500"
    echo "  $0 finetune-all /path/to/checkpoint.pth 500"
    echo "  $0 inference /path/to/checkpoint.pth ./adapter_output_mpiigaze/adapters"
    echo "  $0 inference /path/to/checkpoint.pth ./adapters p03 p06 p12"
    exit 1
}

# 检查参数
if [ $# -lt 1 ]; then
    usage
fi

COMMAND=$1

case $COMMAND in
    test)
        echo -e "${GREEN}Testing MPIIGaze dataset loading...${NC}"
        python test_mpiigaze_dataset.py
        ;;

    train)
        echo -e "${GREEN}Starting Phase 1 training on MPIIGaze...${NC}"
        python train.py --config configs/training/dic_eye_only_mpiigaze.yaml
        ;;

    finetune)
        if [ $# -lt 3 ]; then
            echo -e "${YELLOW}Usage: $0 finetune <subject_id> <checkpoint_path> [max_steps]${NC}"
            echo "Example: $0 finetune p12 /path/to/checkpoint.pth 500"
            exit 1
        fi
        SUBJECT=$2
        CHECKPOINT=$3
        MAX_STEPS=${4:-500}

        echo -e "${GREEN}Fine-tuning adapter for subject: ${SUBJECT}${NC}"
        python finetune_adapter_mpiigaze.py \
            --config configs/training/dic_eye_only_mpiigaze_finetune.yaml \
            --checkpoint ${CHECKPOINT} \
            --subjects ${SUBJECT} \
            --max_steps ${MAX_STEPS}
        ;;

    finetune-all)
        if [ $# -lt 2 ]; then
            echo -e "${YELLOW}Usage: $0 finetune-all <checkpoint_path> [max_steps]${NC}"
            echo "Example: $0 finetune-all /path/to/checkpoint.pth 500"
            exit 1
        fi
        CHECKPOINT=$2
        MAX_STEPS=${3:-500}

        echo -e "${GREEN}Fine-tuning adapters for all subjects (p00-p14)...${NC}"
        python finetune_adapter_mpiigaze.py \
            --config configs/training/dic_eye_only_mpiigaze_finetune.yaml \
            --checkpoint ${CHECKPOINT} \
            --max_steps ${MAX_STEPS}
        ;;

    inference)
        if [ $# -lt 2 ]; then
            echo -e "${YELLOW}Usage: $0 inference <checkpoint_path> [adapter_dir] [subjects...]${NC}"
            echo "Example: $0 inference /path/to/checkpoint.pth ./adapter_output_mpiigaze/adapters"
            echo "Example: $0 inference /path/to/checkpoint.pth ./adapters p03 p06"
            exit 1
        fi
        CHECKPOINT=$2
        ADAPTER_DIR=${3:-""}
        shift 3 2>/dev/null || shift $#
        SUBJECTS=("$@")

        SUBJECTS_ARG=""
        if [ ${#SUBJECTS[@]} -gt 0 ]; then
            SUBJECTS_ARG="--subjects ${SUBJECTS[*]}"
        fi

        if [ -z "$ADAPTER_DIR" ]; then
            echo -e "${GREEN}Running MPIIGaze inference (baseline only, all subjects)...${NC}"
            python inference_mpiigaze.py \
                --config configs/training/dic_eye_only_mpiigaze.yaml \
                --checkpoint ${CHECKPOINT} \
                --output_dir ./inference_mpiigaze \
                ${SUBJECTS_ARG}
        else
            echo -e "${GREEN}Running MPIIGaze inference with adapter comparison...${NC}"
            python inference_mpiigaze.py \
                --config configs/training/dic_eye_only_mpiigaze.yaml \
                --checkpoint ${CHECKPOINT} \
                --adapter ${ADAPTER_DIR} \
                --output_dir ./inference_mpiigaze \
                ${SUBJECTS_ARG}
        fi
        ;;

    *)
        echo -e "${YELLOW}Unknown command: $COMMAND${NC}"
        usage
        ;;
esac

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${BLUE}========================================${NC}"
