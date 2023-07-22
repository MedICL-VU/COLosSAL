#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=PA_brc.stdout
#SBATCH --job-name=PA_brc


module restore deeplearning
source activate seg
cd /scratch/liuh26/src

A=(global global local local)
B=(birch birch birch birch)
C=(5 10 5 10)

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l ${C[$SLURM_ARRAY_TASK_ID]} --plan ${A[$SLURM_ARRAY_TASK_ID]}_${B[$SLURM_ARRAY_TASK_ID]}_${C[$SLURM_ARRAY_TASK_ID]} -n ${A[$SLURM_ARRAY_TASK_ID]}_${B[$SLURM_ARRAY_TASK_ID]}_${C[$SLURM_ARRAY_TASK_ID]}
