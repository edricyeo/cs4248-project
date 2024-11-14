#!/bin/bash
#SBATCH --job-name=HPS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=32gb
#SBATCH --time=15:59:00
#SBATCH --output=logs/%j-slurm.out


source ~/.bashrc

conda activate nlp

# TODO: Insert HF_TOKEN if necessary

nvidia-smi

python ~/cs4248-project/app.py
