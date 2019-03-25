#!/bin/bash

#SBATCH --job-name=c5_delta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --time=03:00:00
#SBATCH --output=c5_delta.out

module load cuda/9.2.88

python -W ignore main.py -o Adadelta -d /scratch/gd66/spring2019/lab2/kaggleamazon/ -w 16 --device GPU