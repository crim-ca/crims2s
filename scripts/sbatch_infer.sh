#!/bin/bash

#SBATCH --job-name=InferS2S
#SBATCH --mail-user=david.landry@crim.ca
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --output=***HOME***listings/%x.out.%j.txt
#SBATCH --gres=gpu:1


CHECKPOINT_FILE="***BASEDIR***/runs/train/multirun/2021-09-13/21-09-57/4/lightning/default_8/0_fb20054ba86747bdb34f53f69f773c5a/checkpoints/epoch\=20-step\=4829.ckpt"

source ***HOME***.bashrc
conda activate s2s

cd ***BASEDIR***/runs/infer
s2s_infer checkpoint_dir=$CHECKPOINT_FILE model=bayes_linear output_file=bayes_linear.nc