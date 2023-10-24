#!/bin/bash

#SBATCH --job-name=HandE_gray_colon_128_256
#SBATCH --partition=normal
#SBATCH --account=conference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hajighasemiamir@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=120:00:00
#SBATCH --error=/home/axh5735/projects/VR/Inference/Inference_H&E/logs/HandE_gray_colon_128_256/slurm_logs/HandE_gray_colon_128_256.err
#SBATCH --output=/home/axh5735/projects/VR/Inference/Inference_H&E/logs/HandE_gray_colon_128_256/slurm_logs/HandE_gray_colon_128_256.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/axh5735/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate IC || fail "conda activate fail"

cd '/home/axh5735/projects/VR/Inference/Inference_H&E'


python '/home/axh5735/projects/VR/Inference/Inference_H&E/inference.py' \
    --patching_seed 2 \
    --patch_size 64 \
    --transformations_read_dir "/home/axh5735/projects/VR/Inference/Inference_H&E/logs/HandE_gray_colon_128_256/transformation" \
    --num_patches_per_image 95 \
    --whitespace_threshold 0.82

