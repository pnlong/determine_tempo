#!/bin/bash
#SBATCH -p free-gpu                     ## run on the gpu partition
#SBATCH -A tdlong_lab_gpu               ## account to charge
#SBATCH -t 2:00                         ## 2-minute run time limit
#SBATCH --job-name=train_tempo_nn       ## pytorch
#SBATCH -N 1                            ## run on a single node
#SBATCH -n 1                            ## request 1 task
#SBATCH --gres=gpu:V100:1               ## request 1 gpu of type V100

# train the neural network on a cluster; request GPU partition

# module load conda and python
module load anaconda/2021.11
module load python/3.11.4

# activate conda env
conda activate

# run python script
python 

# deactivate conda env
conda deactivate