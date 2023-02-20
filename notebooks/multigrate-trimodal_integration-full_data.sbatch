#!/bin/bash

#SBATCH -o multigrate_trimodal_integration-full_data-stdoutput
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1 
#SBATCH -t 2-00:00:00
#SBATCH --mem=90G
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate multigrate
python trimodal_full_integration.py
