#!/bin/bash
#SBATCH --job-name=gauss_seidel_cuda
#SBATCH --output=gauss_seidel_cuda_output.txt
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load cuda
./gauss_seidel_cuda
