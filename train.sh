#!/bin/bash
#SBATCH --job-name=HPS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=32gb
#SBATCH --time=2:59:00
#SBATCH --output=/home/e/e0725981/logs/%j-slurm.out


source ~/.bashrc

conda activate nlp
#conda activate trl2

# TODO: Activate your conda environment
# TODO: Insert HF_TOKEN if necessary

nvidia-smi

#python ~/cs4248-project/app.py
#exit 0

python ~/cs4248-project/inference.py

index=0
for answer_file in "answers"/*; do
    python evaluate-v2.0.py data/dev-v1.1.json "$answer_file" --out-file "eval_${index}.json" --out-image-dir out_images --verbose
    cat "eval_${index}.json"
    ((index++))
done
