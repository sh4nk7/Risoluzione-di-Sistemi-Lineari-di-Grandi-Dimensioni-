#!/bin/bash
#SBATCH --job-name=gauss_seidel_mpi
#SBATCH --output=gauss_seidel_mpi.out
#SBATCH --error=gauss_seidel_mpi.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu

module purge
module load gcc
module load gnu8/8.3.0
module load openmpi4/4.1.1
module load cuda/11.6.0


mpirun --mca opal_warn_on_missing_libcuda 0 ./gauss_seidel_mpi

