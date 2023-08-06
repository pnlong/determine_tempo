#!/bin/bash
#SBATCH -A tdlong_lab                   ## account to charge
#SBATCH -p standard                     ## run on the standard partition
#SBATCH --job-name=gunzip_tempo_data    ## job name
#SBATCH --error=/dfs7/adl/pnlong/artificial_dj/determine_tempo/gunzip_tempo_data.err            ## error log file
#SBATCH --output=/dfs7/adl/pnlong/artificial_dj/determine_tempo/gunzip_tempo_data.out           ## output log file

# README
# Phillip Long
# August 4, 2023
# ungzips and untars the directory created by tempo_dataset.py on the cluster

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"

# tar gzip
# tar -zcvf /Volumes/Seagate/artificial_dj_data/tempo_data.tar.gz /Volumes/Seagate/artificial_dj_data/tempo_data

tar -xvzf "${data}/tempo_data.tar.gz" -C "${data}/tempo_data"

