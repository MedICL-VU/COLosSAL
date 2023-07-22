# ColdStartAL
Cold start active learning for 3D medical image segmentation

### What is cold start active learning and why do we care?
For multi-organ segmentation task, there are two main directions of research: (1) for the organs whose annotations are publicly available such as liver and kidney (e.g., LiTS, KiTS), how do we train a single unified model to segment all these organs? (2) for the organs whose annotations need to be created from the ground up, how can we annotate data more efficiently and how to quickly achieve a decent performance with only one or a few annotation(s)? Cold start active learning aims to answer the second question from a perspective of "how to select the most important data for the initial annotation from a large unlabeled data pool". Cold start means that we start with ZERO labels for the target organ, and the data selection strategy is completely independent of label information. For example, imagine we are running a start-up company and we aim to train a segmentation model for organ X. Organ X has never been segmented by any other companies and we hope to create a labeled dataset for this organ from ground up. Now, we know that we have collected 100K images that include organ X, but we do not want to spend a crazy amount of money to have experts annotate all 100k data (annotators also hate it!). The question is, can we select the most "important" data for annotation so that reasonable segmentation performance for organ X can be achieved quickly? Of course, once resonable performance is achieved, we can use semi-supervised learning or other techniques to further boost the performance, without human annotations.

### What has been explored and what not?
For warm start active learning, uncertainty-based and diversity-based selection strategies are the most widely used methods. Uncertainty-based methods assume that labeling the most uncertain data and training the network with them can effectively improve the model performance. Diversity-based methods assume that some of the training data can be very similar, and more information can be learned by the network if the labeled data are distinct from each other. However, do these techniques also work for cold start scenarios? We don't know yet...

### Dependencies
```shell script
CUDA 11.4
cudnn 8.5.0
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Training baselines (random selection)
```shell script
python -W ignore run_training.py --nid 1 --dropout 0.2 -o Heart -c 2 -m mr -l 5 --RS_seed 0 -n random_num_5_seed_0
```

where
* --nid: network id. Please use 1 (3D U-Net from MONAI) to make sure all experiments are conducted using the same network architecture.
* --dropout: dropout ratio. Use dropout ratio=0.2.
* -o: The name of the organ. E.g., Spleen, Liver, Pancreas, etc.
* -c: The number of classes/output channels. For single-organ segmentation, use c=2.
* -m: Modality. ['ct' | 'mr']
* -l: The number of labeled data. 
* --RS_seed: The seed for random selection of training data.
* -n: The name of the experiment.

### TODO
* update readme with training w/ uncertainty/diversity-based selections
