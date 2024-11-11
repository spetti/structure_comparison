#!/bin/bash
#SBATCH --job-name=Tmain
#SBATCH --partition=mpi
#SBATCH --mem=16G
#SBATCH --output=Tmain.out
#SBATCH --error=Tmain.err
#SBATCH --time=24:00:00

config_file_name=$1

source activate benchenv 
module load cuda/12.2

snakemake --rerun-incomplete --cores all --jobs 150 --restart-times 1 --latency-wait 60\
  --config config_file_name="$config_file_name" \
  --cluster "sbatch --partition=mpi,batch,preempt --mem={resources.mem_mb} -c {resources.cpus} --time={resources.time} --output='$config_file_name'_slurm_output/%j.out --error='$config_file_name'_slurm_output/%j.err" \
  --jobname "{wildcards}_{jobid}" \
  --printshellcmds
