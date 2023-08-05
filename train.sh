#!/bin/bash
#SBATCH --job-name=train_tempo_nn       ## job name
#SBATCH -A tdlong_lab_gpu               ## account to charge
#SBATCH -p free-gpu                     ## run on the gpu partition
#SBATCH --nodes=1                       ## run on a single node
#SBATCH --ntasks=1                      ## request 1 task
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH -t 2:00                         ## 2-minute run time limit
#SBATCH --gres=gpu:V100:1               ## request 1 gpu of type V100
#SBATCH --error=/dfs7/adl/pnlong/artificial_dj/determine_tempo/train_tempo_nn.err            ## error log file
#SBATCH --output=/dfs7/adl/pnlong/artificial_dj/determine_tempo/train_tempo_nn.out           ## output log file

# train the neural network on a cluster; request GPU partition

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"

# command to replace filepaths in data file
# sed "s+/Volumes/Seagate/artificial_dj_data+${data}+g" "${data}/tempo_data.tsv" > "${data}/tempo_data.cluster.tsv"

# module load conda and python
module load anaconda/2022.05
module load python/3.10.2

# activate conda env
source /data/homezvol2/pnlong/.condarc
conda activate "${artificial_dj}/envs"

# run python script
python "${artificial_dj}/determine_tempo/tempo_neural_network.py" "${data}/tempo_data.cluster.tsv" "${data}/tempo_nn.pth"
