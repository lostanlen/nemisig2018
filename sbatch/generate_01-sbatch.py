import os
import sys

sys.path.append("../src")
import localmodule


# Define constants.
data_dir = localmodule.get_data_dir()
dataset_name = localmodule.get_dataset_name()
composers = localmodule.get_composers()
kern_name = "_".join([dataset_name, "kern"])
kern_dir = os.path.join(data_dir, kern_name)
script_name = "01_eigenprogression_transform.py"
script_path = os.path.join("..", "..", "..", "src", script_name)


# Create folders.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)


# Loop over composers.
for composer_str in composers:

    composer_dir = os.path.join(kern_dir, composer_str)
    pieces = os.listdir(composer_dir)

    # Loop over pieces.
    for piece_name in pieces:

        piece_str = piece_name[:-4]
        job_name = "_".join([
            script_name[:2], composer_str, piece_str])
        file_name = job_name + ".sbatch"
        file_path = os.path.join(sbatch_dir, file_name)

        # Define script.
        script_list = [script_path, composer_str, piece_str]
        script_path_with_args = " ".join(script_list)

        # Open file.
        with open(file_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write("#BATCH --job-name=" + job_name + "\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --tasks-per-node=1\n")
            f.write("#SBATCH --cpus-per-task=1\n")
            f.write("#SBATCH --time=4:00:00\n")
            f.write("#SBATCH --mem=128GB\n")
            f.write("#SBATCH --output=../slurm/slurm_" + job_name + "_%j.out\n")
            f.write("\n")
            f.write("module purge\n")
            f.write("\n")
            f.write("# The first argument is the name of the composer.\n")
            f.write("# The second argument is the name of the piece.\n")
            f.write("python " + script_path_with_args)
