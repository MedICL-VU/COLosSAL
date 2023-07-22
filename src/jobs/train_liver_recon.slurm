#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=liver_recon_%A_%a.stdout
#SBATCH --job-name=liver_recon


module restore deeplearning
source activate seg
cd /scratch/liuh26/src


python -W ignore run_training_recon.py --nid 1 --dropout 0.2 -o Liver -c 1 -n recon_ae