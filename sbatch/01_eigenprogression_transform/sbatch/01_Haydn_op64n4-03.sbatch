#!/bin/bash

#BATCH --job-name=01_Haydn_op64n4-03
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/slurm_01_Haydn_op64n4-03_%j.out

module purge

# The first argument is the name of the composer.
# The second argument is the name of the piece.
python ../../../src/01_eigenprogression_transform.py Haydn op64n4-03