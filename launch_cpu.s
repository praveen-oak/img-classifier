#!/bin/bash

#SBATCH --job-name=lab2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --time=01:00:00
#SBATCH --partition=c32_41
#SBATCH --output=out.%j

python main.py -o SGD -d /scratch/gd66/spring2019/lab2/kaggleamazon/ -w 1

