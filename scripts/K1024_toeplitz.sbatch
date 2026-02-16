#!/bin/bash
#SBATCH --job-name=GKW_K1024
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --output=data/K1024_slurm_%j.log
#SBATCH --error=data/K1024_slurm_%j.err

# GKW spectral computation at K=1024, P=2048 (single-threaded).
# Estimated: 25-41 hours, ~10GB peak RAM.
#
# Setup (run once before submitting):
#   module load julia/1.11.3
#   cd /path/to/GKWExperiments.jl
#   julia --project -e 'using Pkg; Pkg.instantiate()'
#
# Submit:
#   sbatch scripts/K1024_toeplitz.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f data/K1024_slurm_*.log

module load julia/1.11.3

cd $SLURM_SUBMIT_DIR

julia --project="." --startup-file=no -t 1 scripts/bigfloat_spectral_K1024.jl
