#!/bin/bash
#SBATCH --job-name=main
#SBATCH --partition=mpi
#SBATCH --mem=1G
#SBATCH --output=main.out
#SBATCH --error=main.err
#SBATCH --time=24:00:00

input_dir=$1

source activate benchenv 
module load cuda/12.2

snakemake -s Snakefile_compute_BLOSUM_ect --rerun-incomplete --cores all --jobs 150 --restart-times 0 --latency-wait 60 \
  --config input_dir="$input_dir" \
  --cluster "sbatch --partition=pettilab --mem={resources.mem_mb} -c {resources.cpus} --time={resources.time}  --gres=gpu:{resources.gpus} --output='$input_dir'_slurm_output/%j.out --error='$input_dir'_slurm_output/%j.err" \
  --jobname "{jobid}" \
  --printshellcmds


