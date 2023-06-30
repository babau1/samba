#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=old
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=16G
#SBATCH --account=def-corbeilj
#SBATCH --output=old_output.out
source ~/dev_expes/bin/activate
python ~/summit/summit/execute.py --config_path "/home/bbauvin/projects/def-corbeilj/bbauvin/datasets/config_file_old.yml"