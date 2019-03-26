#!/bin/bash

#SBATCH --job-name=c2_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --time=06:00:00
#SBATCH --partition=knl
#SBATCH --output=c2_1_knl.out

python -W ignore main.py -o SGD -d /scratch/gd66/spring2019/lab2/kaggleamazon/ -w 1 --precision true

