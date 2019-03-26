#!/bin/bash

#SBATCH --job-name=c5_adam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --time=01:00:00
#SBATCH --output=c5_adam.out

module load cuda/9.2.88

python -W ignore main.py -d /scratch/gd66/spring2019/lab2/kaggleamazon/ -w 20 --device GPU -o Adam --precision true
