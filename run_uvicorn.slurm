#!/bin/bash
#SBATCH -p sm
#SBATCH --qos=priority
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -J uvicorn_server
#SBATCH -o uvicorn_server.out
#SBATCH -e uvicorn_server.err

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate search2

# Go to the repo
cd /data/sls/scratch/akolli/search_similarity

# Run server
python3 -m uvicorn search_endpoint:app --host 0.0.0.0 --port 8000
