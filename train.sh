#!/bin/bash
#SBATCH --job-name=TRL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=32gb
#SBATCH --time=00:59:00
#SBATCH --output=/home/e/e0725981/logs/%j-slurm.out

# Load the Conda environment
source ~/.bashrc
#conda init



nvidia-smi

#python ~/cs4248-project/app.py

python ~/cs4248-project/inference.py
python evaluate-v2.0.py data/dev-v1.1.json answers.json --out-file eval.json --out-image-dir out_images --verbose
