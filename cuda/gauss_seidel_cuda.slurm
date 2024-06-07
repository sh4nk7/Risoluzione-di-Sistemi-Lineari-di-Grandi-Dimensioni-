#!/bin/bash
#SBATCH --job-name=gauss_seidel_cuda
#SBATCH --output=gauss_seidel_cuda_output.txt
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

module load cuda

nvcc -o gauss_seidel_cuda gauss_seidel_cuda.cu
./gauss_seidel_cuda
