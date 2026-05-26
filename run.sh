#!/bin/bash
# =============================================================================
# X-Former 训练启动脚本
# 支持：单卡训练 / 多卡 DDP / nohup 后台运行
#
# 消融实验：修改 config/config.yaml 中 ablation 部分的 true/false 即可
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="config/config.yaml"
USE_DDP=false
USE_NOHUP=false
GPU_IDS="0"

print_usage() {
    cat << 'EOF'
用法: bash run.sh [选项]

选项:
  --config FILE    配置文件路径 (默认: config/config.yaml)
  --ddp            启用多卡 DistributedDataParallel
  --gpus "0,1,2"   指定 GPU (默认: 0)
  --nohup          后台运行，关终端不中断

示例:
  bash run.sh                          # 单卡 GPU 0 训练
  bash run.sh --ddp --nohup            # 3 卡 DDP 后台训练
  bash run.sh --gpus "1" --nohup       # GPU 1 后台训练
  bash run.sh --ddp --nohup --gpus "0,1,2"

消融实验：编辑 config/config.yaml 中 ablation 部分:
  use_bottleneck: false       # 去掉信息瓶颈
  use_perceiver: false        # 去掉 Perceiver
  use_domain_loss: false      # 去掉域对抗损失
  use_contrastive_loss: false # 去掉对比损失
  use_consistency_loss: false # 去掉一致性损失
  use_lesion: false           # 去掉病灶影像组学
  use_proto_clinical: false   # 用 MLP 替代原型编码器
  joint_training: false       # 独立训练两个数据集
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2"; shift 2 ;;
        --ddp)      USE_DDP=true; shift ;;
        --gpus)     GPU_IDS="$2"; shift 2 ;;
        --nohup)    USE_NOHUP=true; shift ;;
        --help|-h)  print_usage ;;
        *)          echo "未知参数: $1"; print_usage ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="xformer_${TIMESTAMP}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if $USE_DDP; then
    N_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -u train/crossval_joint.py --config $CONFIG --ddp"
else
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -u train/crossval_joint.py --config $CONFIG"
fi

echo "============================================"
echo "X-Former Training"
echo "============================================"
echo "运行名称: $RUN_NAME"
echo "GPU:      $GPU_IDS"
echo "DDP:      $USE_DDP"
echo "配置:     $CONFIG"
echo "日志:     $LOG_FILE"
echo "命令:     $CMD"
echo "============================================"

if $USE_NOHUP; then
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "已后台启动 (PID: $PID)"
    echo ""
    echo "监控: tail -f $LOG_FILE"
    echo "GPU:  watch -n 1 nvidia-smi"
    echo "终止: kill $PID"
else
    echo "开始训练... (Ctrl+C 中断)"
    echo ""
    bash -c "$CMD" 2>&1 | tee "$LOG_FILE"
fi
