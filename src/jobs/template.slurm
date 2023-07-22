#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=5-10:5
#SBATCH --output=spleen_proxy_%A_%a.stdout
#SBATCH --job-name=weak_variance


module restore deeplearning
source activate seg
cd /scratch/liuh26/src


python -W ignore run_training.py --nid 1 --dropout 0.2 -o Spleen -c 2 -l ${SLURM_ARRAY_TASK_ID} -n WeakVariance_num_${SLURM_ARRAY_TASK_ID} --plan_name WeakAug_mc_20_variance