#!/bin/bash
# FAGE MPIIGaze Adapter Fine-tuning Script
# 为 MPIIGaze 数据集的每个 subject 生成个性化 adapter 权重

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
CONFIG="configs/training/dic_eye_only_mpiigaze_finetune.yaml"
OUTPUT_DIR="./adapter_output_mpiigaze"
MAX_STEPS=500
LR=5e-4
BATCH_SIZE=8
LPIPS_WEIGHT=0.1
ID_WEIGHT=0.1
PRETRAIN_STEPS=0  # 可选：val-set 预训练步数

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}========================================"
    echo "  FAGE MPIIGaze Adapter Fine-tuning"
    echo -e "========================================${NC}"
}

print_usage() {
    cat << EOF
用法: $0 <command> [args]

命令:
  single <subject> <checkpoint> [steps]
      为单个 subject 训练 adapter
      示例: $0 single p12 /path/to/checkpoint.pth 500

  val <checkpoint> [steps]
      为所有 val subjects 训练 adapters
      示例: $0 val /path/to/checkpoint.pth 500

  test <checkpoint> [steps]
      为所有 test subjects 训练 adapters
      示例: $0 test /path/to/checkpoint.pth 500

  all <checkpoint> [steps]
      为所有 subjects (train+val+test) 训练 adapters
      示例: $0 all /path/to/checkpoint.pth 500

  batch <checkpoint> <subject1> <subject2> ... [steps]
      为指定的多个 subjects 训练 adapters
      示例: $0 batch /path/to/checkpoint.pth p12 p03 p06 500

参数:
  checkpoint    Phase 1 预训练模型路径
  subject       Subject ID (e.g., p12, p03)
  steps         训练步数 (默认: 500)

环境变量:
  PRETRAIN_STEPS    Val-set 预训练步数 (默认: 0, 禁用)
  LPIPS_WEIGHT      LPIPS 损失权重 (默认: 0.1)
  ID_WEIGHT         ArcFace ID 损失权重 (默认: 0.1)
  BATCH_SIZE        批大小 (默认: 8)
  LR                学习率 (默认: 5e-4)

示例:
  # 单个用户
  $0 single p12 /path/to/checkpoint.pth

  # 所有 val 用户
  $0 val /path/to/checkpoint.pth

  # 所有 test 用户（带预训练）
  PRETRAIN_STEPS=1000 $0 test /path/to/checkpoint.pth

  # 自定义参数
  BATCH_SIZE=16 LR=1e-3 $0 single p12 /path/to/checkpoint.pth 1000
EOF
}

if [ $# -lt 1 ]; then
    print_header
    print_usage
    exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
    single)
        if [ $# -lt 2 ]; then
            echo -e "${RED}错误: 需要 subject 和 checkpoint 参数${NC}"
            print_usage
            exit 1
        fi
        SUBJECT=$1
        CHECKPOINT=$2
        STEPS=${3:-$MAX_STEPS}

        print_header
        echo "模式: 单用户微调"
        echo "Subject: $SUBJECT"
        echo "Checkpoint: $CHECKPOINT"
        echo "Steps: $STEPS"
        echo "Output: $OUTPUT_DIR/adapters/${SUBJECT}.pth"
        echo "========================================"

        python finetune_adapter_mpiigaze.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --subjects "$SUBJECT" \
            --max_steps "$STEPS" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --lpips_weight "$LPIPS_WEIGHT" \
            --id_weight "$ID_WEIGHT" \
            --pretrain_steps "$PRETRAIN_STEPS" \
            --output_dir "$OUTPUT_DIR"
        ;;

    val)
        if [ $# -lt 1 ]; then
            echo -e "${RED}错误: 需要 checkpoint 参数${NC}"
            print_usage
            exit 1
        fi
        CHECKPOINT=$1
        STEPS=${2:-$MAX_STEPS}

        print_header
        echo "模式: Val split 批量微调"
        echo "Checkpoint: $CHECKPOINT"
        echo "Steps per subject: $STEPS"
        echo "Pretrain steps: $PRETRAIN_STEPS"
        echo "Output: $OUTPUT_DIR/adapters/"
        echo "========================================"

        python finetune_adapter_mpiigaze.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --split val \
            --max_steps "$STEPS" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --lpips_weight "$LPIPS_WEIGHT" \
            --id_weight "$ID_WEIGHT" \
            --pretrain_steps "$PRETRAIN_STEPS" \
            --output_dir "$OUTPUT_DIR"
        ;;

    test)
        if [ $# -lt 1 ]; then
            echo -e "${RED}错误: 需要 checkpoint 参数${NC}"
            print_usage
            exit 1
        fi
        CHECKPOINT=$1
        STEPS=${2:-$MAX_STEPS}

        print_header
        echo "模式: Test split 批量微调"
        echo "Checkpoint: $CHECKPOINT"
        echo "Steps per subject: $STEPS"
        echo "Pretrain steps: $PRETRAIN_STEPS"
        echo "Output: $OUTPUT_DIR/adapters/"
        echo "========================================"

        python finetune_adapter_mpiigaze.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --split test \
            --max_steps "$STEPS" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --lpips_weight "$LPIPS_WEIGHT" \
            --id_weight "$ID_WEIGHT" \
            --pretrain_steps "$PRETRAIN_STEPS" \
            --output_dir "$OUTPUT_DIR"
        ;;

    all)
        if [ $# -lt 1 ]; then
            echo -e "${RED}错误: 需要 checkpoint 参数${NC}"
            print_usage
            exit 1
        fi
        CHECKPOINT=$1
        STEPS=${2:-$MAX_STEPS}

        print_header
        echo "模式: 全部 subjects 批量微调"
        echo "Checkpoint: $CHECKPOINT"
        echo "Steps per subject: $STEPS"
        echo "Pretrain steps: $PRETRAIN_STEPS"
        echo "========================================"

        for SPLIT in train val test; do
            echo -e "${YELLOW}Processing $SPLIT split...${NC}"
            python finetune_adapter_mpiigaze.py \
                --config "$CONFIG" \
                --checkpoint "$CHECKPOINT" \
                --split "$SPLIT" \
                --max_steps "$STEPS" \
                --lr "$LR" \
                --batch_size "$BATCH_SIZE" \
                --lpips_weight "$LPIPS_WEIGHT" \
                --id_weight "$ID_WEIGHT" \
                --pretrain_steps "$PRETRAIN_STEPS" \
                --output_dir "$OUTPUT_DIR"
        done
        ;;

    batch)
        if [ $# -lt 2 ]; then
            echo -e "${RED}错误: 需要 checkpoint 和至少一个 subject${NC}"
            print_usage
            exit 1
        fi
        CHECKPOINT=$1
        shift

        # 提取 subjects 和可选的 steps
        SUBJECTS=()
        while [ $# -gt 0 ]; do
            if [[ $1 =~ ^[0-9]+$ ]]; then
                STEPS=$1
                break
            else
                SUBJECTS+=("$1")
            fi
            shift
        done

        if [ ${#SUBJECTS[@]} -eq 0 ]; then
            echo -e "${RED}错误: 未指定 subjects${NC}"
            exit 1
        fi

        STEPS=${STEPS:-$MAX_STEPS}

        print_header
        echo "模式: 批量微调指定 subjects"
        echo "Checkpoint: $CHECKPOINT"
        echo "Subjects: ${SUBJECTS[*]}"
        echo "Steps per subject: $STEPS"
        echo "Output: $OUTPUT_DIR/adapters/"
        echo "========================================"

        python finetune_adapter_mpiigaze.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --subjects "${SUBJECTS[@]}" \
            --max_steps "$STEPS" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --lpips_weight "$LPIPS_WEIGHT" \
            --id_weight "$ID_WEIGHT" \
            --pretrain_steps "$PRETRAIN_STEPS" \
            --output_dir "$OUTPUT_DIR"
        ;;

    *)
        echo -e "${RED}错误: 未知命令 '$COMMAND'${NC}"
        print_usage
        exit 1
        ;;
esac

echo -e "${GREEN}========================================"
echo "Done!"
echo -e "========================================${NC}"
