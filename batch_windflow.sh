#!/bin/bash

#SBATCH --cpus-per-task=4 # Please keep to <= 16 per gpu requested, though you are unlikely to get improvements beyond _4_.
#SBATCH --mem=16000 # MB of system memory. I never seem to need more than 8000. Please keep to <=190000.
#SBATCH -o /home/pscudder/sb_log/sb_%j.log
#SBATCH -p salvador
#SBATCH -t 22:00:00
#SBATCH --gpus=1
#SBATCH --exclude=gustav
#SBATCH --nodelist r740-105-15

# Disable output buffering
export PYTHONUNBUFFERED=1

# Set environment
source ~/.bash_profile
conda activate pytorch_windflow

# Print some diagnostic information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "As user: $(whoami)"
echo "With GPU: $(nvidia-smi -L)"

# Run script with unbuffered output
python -u train.py \
    --model_path models/raft-size_512/ \
    --dataset g5nr \
    --data_path /ships19/cryo/daves/windflow/paul/G5NR_patches \
    --model_name raft \
    --batch_size 2 \
    --loss L1 \
    --max_iterations 500 \
    --lr 0.00001 \
    --log_step 10

echo "Job finished at $(date)"

#salloc -w r740-105-15 -p salvador