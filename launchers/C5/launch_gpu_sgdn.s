#!/bin/bash

#SBATCH --job-name=c5_sgdn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --time=01:00:00
#SBATCH --output=c5_sgdn.out

module load cuda/9.2.88

python -W ignore main.py -d /scratch/gd66/spring2019/lab2/kaggleamazon/ -w 20 --device GPU -o SGDN --precision true
