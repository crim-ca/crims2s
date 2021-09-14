#!/bin/bash

#SBATCH --job-name=InferS2S
#SBATCH --mail-user=david.landry@crim.ca
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --output=***HOME***listings/%x.out.%j.txt
#SBATCH --gres=gpu:1


CHECKPOINT_FILE="***BASEDIR***/runs/multirun/2021-09-14/10-56-58/0/lightning/default_8/0_3f62a5ca126c49538a0dc36c20f1a77f/checkpoints/epoch\=39-step\=36719.ckpt"

source ***HOME***.bashrc
conda activate s2s

cd ***BASEDIR***/runs/infer
s2s_infer checkpoint_dir=$CHECKPOINT_FILE model=emos_multiplexed_normalcubenormal transform=emos_cube output_file=emos_cube.nc