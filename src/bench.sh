#!/bin/bash
#SBATCH --job-name=chiplet-fence
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --output=chiplet-fence-%j.out
#SBATCH --error=chiplet-fence-%j.err

# Context for the log
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Run
hipcc bench.hip
srun ./a.out