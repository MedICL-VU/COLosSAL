#!/usr/bin/env bash

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Spleen -c 2 -m ct -l 5 --plan global_typiclust_5 -n global_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Spleen -c 2 -m ct -l 5 --plan local_typiclust_5 -n local_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Spleen -c 2 -m ct -l 10 --plan global_typiclust_10 -n global_typiclust_10
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Spleen -c 2 -m ct -l 10 --plan local_typiclust_10 -n local_typiclust_10

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 5 --plan global_typiclust_5 -n global_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 5 --plan local_typiclust_5 -n local_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 10 --plan global_typiclust_10 -n global_typiclust_10
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 10 --plan local_typiclust_10 -n local_typiclust_10

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 3 --plan global_typiclust_3 -n global_typiclust_3
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 3 --plan local_typiclust_3 -n local_typiclust_3
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 5 --plan global_typiclust_5 -n global_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 5 --plan local_typiclust_5 -n local_typiclust_5

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 5 --plan global_typiclust_5 -n global_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 5 --plan local_typiclust_5 -n local_typiclust_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 10 --plan global_typiclust_10 -n global_typiclust_10
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 10 --plan local_typiclust_10 -n local_typiclust_10

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 5 --plan global_birch_5 -n global_birch_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 5 --plan local_birch_5 -n local_birch_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 10 --plan global_birch_10 -n global_birch_10
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Liver -c 3 -m ct -l 10 --plan local_birch_10 -n local_birch_10

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 3 --plan global_birch_3 -n global_birch_3
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 3 --plan local_birch_3 -n local_birch_3
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 5 --plan global_birch_5 -n global_birch_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 5 --plan local_birch_5 -n local_birch_5

python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 5 --plan global_birch_5 -n global_birch_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 5 --plan local_birch_5 -n local_birch_5
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 10 --plan global_birch_10 -n global_birch_10
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Pancreas -c 3 -m ct -l 10 --plan local_birch_10 -n local_birch_10
