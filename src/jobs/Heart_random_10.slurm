#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --output=Heart_10_%a.stdout
#SBATCH --job-name=Heart_10


module restore deeplearning
source activate seg
cd /scratch/liuh26/src


python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 10 -n random_num_10_seed_${SLURM_ARRAY_TASK_ID} --RS_seed ${SLURM_ARRAY_TASK_ID}