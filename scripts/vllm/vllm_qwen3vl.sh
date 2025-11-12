#!/bin/bash
#SBATCH --partition=GPU-4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=/home/yaoyhu/data/rsstvlm/logs/qwen-3vl-%j.out


# 1. 推荐在开头结尾打印时间及基本信息
echo "=========================================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node list: $SLURM_JOB_NODELIST"
echo "=========================================================="
echo ""

# 2. 激活目标环境
source ~/.bashrc
cd ~/data/rsstvlm/
echo "pwd: $(pwd)"
echo "which Python: $(uv run which python)"

# 3. 运行您的 Python 脚本
echo "🤖 Deploying Qwen/Qwen3-VL-30B-A3B-Instruct via vllm..."
echo

<<<<<<< HEAD
uv run vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --served-model-name qwen3-vl-30b \
    --swap-space 16 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 8 \
    --enforce-eager \
=======
export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --served-model-name qwen3-vl-30b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --swap-space 32 \
>>>>>>> 2965eaf (build: update CUDA to 12.8 for Qwen3-VL deployment)
    --host 0.0.0.0 \
    --port 8002

# 4. 打印作业结束时间
echo ""
echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
