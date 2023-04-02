#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q long
#$ -e errors/
#$ -N gen_sif_icwsm

# Required modules
module load conda
conda init bash
source activate sif

python sif_embedding.py