#!/bin/bash
#SBATCH -p free-gpu                     ## run on the gpu partition
#SBATCH -A tdlong_lab_gpu               ## account to charge
#SBATCH -t 2:00                         ## 2-minute run time limit
#SBATCH --job-name=train_tempo_nn       ## pytorch
#SBATCH -N 1                            ## run on a single node
#SBATCH -n 1                            ## request 1 task
#SBATCH --gres=gpu:V100:1               ## request 1 gpu of type V100

# train the neural network on a cluster; request GPU partition

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"

# command to replace filepaths in data file
# sed "s+/Volumes/Seagate/artificial_dj_data+${data}+g" "${data}/tempo_data.tsv" > "${data}/tempo_data.cluster.tsv"

# module load conda and python
module load anaconda/2022.05
module load python/3.10.2

# activate conda env
conda activate "${artificial_dj}/envs"

# run python script
python "${artificial_dj}/determine_tempo/tempo_neural_network.py" "${data}/tempo_data.cluster.tsv" "${data}/tempo_nn.pth"

# deactivate conda env
conda deactivate