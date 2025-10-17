#!/bin/bash
#SBATCH --partition=GPU-4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=/home/yaoyhu/data/rsstvlm/logs/download_papers-%j.out


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
echo "Running"
echo
uv run rsstvlm/rag/paper_download.py

# 4. 打印作业结束时间
echo ""
echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="