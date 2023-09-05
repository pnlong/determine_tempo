#!/bin/bash
#SBATCH --job-name=test_tempo_nn        ## job name
#SBATCH -A tdlong_lab                   ## account to charge
#SBATCH -p standard                     ## run on the standard cpu partition
#SBATCH --nodes=1                       ## run on a single node
#SBATCH --ntasks=1                      ## request 1 task
#SBATCH --cpus-per-task=1               ## number of cores the job needs

# README
# Phillip Long
# August 20, 2023
# script to test the neural network; request CPU partition
# assumes I have already run tempo_dataset.py and tempo_neural_network.py

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"
output_prefix="${data}/tempo_nn"

# module load conda (hpc3 help says not to load python + conda together)
module load miniconda3/4.12.0

# activate conda env
eval "$(/opt/apps/miniconda3/4.12.0/bin/conda 'shell.bash' 'hook')"
conda activate artificial_dj

# run python script
python "${artificial_dj}/determine_tempo/tempo_inferences.py" "${data}/tempo_data.cluster.tsv" "${output_prefix}.pth"
