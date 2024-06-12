#!/bin/bash
#SBATCH --job-name=gauss_seidel_cuda
#SBATCH --output=gauss_seidel_cuda.out
#SBATCH --error=gauss_seidel_cuda.err
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu        # Modifica questa riga per specificare la partizione GPU corretta
#SBATCH --qos=gpu              # Modifica questa riga per specificare il QoS GPU corretto

module load gcc
module load gnu8/8.3.0
module load openmpi4/4.1.1  
module load cuda/11.6.0


srun ./gauss_seidel_cuda matrix.txt vector.txt
