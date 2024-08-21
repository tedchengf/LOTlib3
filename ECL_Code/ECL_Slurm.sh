#!/bin/bash

#SBATCH --job-name=ECL_B1           # The name of the job
#SBATCH --nodes=1                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=20             # Request 1 CPU per job instance
#SBATCH --mem=300GB                     # Request 2GB of RAM per job instance
#SBATCH --time=04:00:00               # Request 10 mins per job instance
#SBATCH --output=/vast/fc1367/ECL/ECL_Code/output/out_%A_%a.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=fc1367@nyu.edu   # Email address
#SBATCH --mail-type=END               # Send an email when all the instances of this job are completed

/vast/fc1367/ECL/ECL_Code/Run_ECL.sh conda activate python3 /vast/fc1367/ECL/ECL_Code/Filter_Driver.py