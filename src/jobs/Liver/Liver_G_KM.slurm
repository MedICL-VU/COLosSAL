#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=LVR_G_KM.stdout
#SBATCH --job-name=LVR_G_KM


module restore deeplearning
source activate seg
cd /scratch/liuh26/src


python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 5 --plan kmeans_global_5 -n global_km