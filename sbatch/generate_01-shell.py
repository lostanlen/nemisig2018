import numpy as np
import os
import sys

sys.path.append("../src")
import localmodule


# Define constants
data_dir = localmodule.get_data_dir()
dataset_name = localmodule.get_dataset_name()
composers = localmodule.get_composers()
kern_name = "_".join([dataset_name, "kern"])
kern_dir = os.path.join(data_dir, kern_name)
script_name = "01_eigenprogression-transform.py"
script_path = os.path.join("..", "src", script_name)


# Create folders.
os.makedirs(script_name[:-3], exist_ok=True)
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)


# Loop over composers.
for composer_str in composers:

    file_name = script_name[:3] + "_" + composer_str + ".sh"
    file_path = os.path.join(sbatch_dir, file_name)

    # Open shell file
    with open(file_path, "w") as f:
        # Print header
        f.write(
            "# This shell script executes Slurm jobs for computing\n" +
            "# eigenprogression transforms\n" +
            "# on " + dataset_name + ".\n")
        f.write("# Composer: " + composer_str + ".\n")
        f.write("\n")

        composer_dir = os.path.join(kern_dir, composer_str)
        pieces = os.listdir(composer_dir)

        # Loop over pieces
        for piece_name in pieces:
            # Define job name.
            piece_str = piece_name[:-4]
            job_name = "_".join(
                [script_name[:3], composer_str, piece_str])
            sbatch_str = "sbatch " + job_name + ".sbatch"

            # Write SBATCH command to shell file.
            f.write(sbatch_str + "\n")

        # Grant permission to execute the shell file.
        # https://stackoverflow.com/a/30463972
        mode = os.stat(file_path).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(file_path, mode)


# Create meta-shell script.
file_name = script_name[:3] + ".sh"
file_path = os.path.join(sbatch_dir, file_name)

# Open shell file
with open(file_path, "w") as f:
    # Print header
    f.write(
        "# This shell script executes Slurm jobs for computing\n" +
        "# eigenprogression transforms\n" +
        "# on " + dataset_name + ".\n")
    f.write("# Composers: " + ", ".join(composers) + ".\n")
    f.write("\n")

    # Loop over composers.
    for composer_str in composers:
        shell_name = "./" + script_name[:3] + "_" + composer_str + ".sh"
        f.write(shell_name + "\n")

    f.write("\n")
