#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=Heart_Recon.stdout
#SBATCH --job-name=Heart_Recon


module restore deeplearning
source activate seg
cd /scratch/liuh26/src


python -W ignore run_training_recon.py --nid 1 --dropout 0.2 -c 1 -o Heart -m mr -n recon