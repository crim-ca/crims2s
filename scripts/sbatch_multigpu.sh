#!/bin/bash

#SBATCH --job-name=TrainS2S
#SBATCH --mail-user=david.landry@crim.ca
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --output=***HOME***listings/%x.out.%j.txt
#SBATCH --gres=gpu:2

source ***HOME***.bashrc
conda activate s2s

s2s_train experiment=conv backend=multigpu backend.gpus=2 experiment.batch_size=1