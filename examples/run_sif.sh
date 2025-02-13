#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q long
#$ -e errors/
#$ -N gen_icwsm_caption_sif

# Required modules
module load conda
conda init bash
source activate sif

python sif_embedding.py