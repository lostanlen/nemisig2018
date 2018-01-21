#!/bin/bash

#BATCH --job-name=01_Haydn_op20n6-02
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --output=../slurm/slurm_01_Haydn_op20n6-02_%j.out

module purge

# The first argument is the name of the composer.
# The second argument is the name of the piece.
python ../../../src/01_eigenprogression_transform.py Haydn op20n6-02