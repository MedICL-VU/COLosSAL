#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=HRT_L_ENT.stdout
#SBATCH --job-name=HRT_L_ENT


module restore deeplearning
source activate seg
cd /scratch/liuh26/src


python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 5 --plan local_mc_20_entropy -n local_entropy