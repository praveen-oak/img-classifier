#!/bin/bash

#SBATCH --job-name=c3_20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --time=03:00:00
#SBATCH --partition=knl
#SBATCH --output=c3_20_knl.out

python -W ignore -m cProfile -o c3_20_knl.prof main.py -o SGD -d /scratch/gd66/spring2019/lab2/kaggleamazon/ -w 20