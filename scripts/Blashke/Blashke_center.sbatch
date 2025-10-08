#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu

# Makes the julia binary available. This will depend on your HPC setup
module load julia/1.11.3

# Run the job with full optimisations. Add in --project="." if you have a
# Project/Manifest with dependencies
julia --project="." run_Blashke_center.jl
