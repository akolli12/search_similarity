#!/bin/bash
#SBATCH -p sm
#SBATCH --qos=priority
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -J uvicorn_server
#SBATCH -o uvicorn_server.out
#SBATCH -e uvicorn_server.err

# --- Fix conda on CSAIL Slurm ---
export CONDA_ENVS_PATH=/data/sls/scratch/akolli/miniconda3/envs
export CONDA_PKGS_DIRS=/data/sls/scratch/akolli/miniconda3/pkgs
export CONDA_CONFIG=/dev/null
export CONDA_NO_PLUGINS=true

# --- Initialize conda ---
source /data/sls/scratch/akolli/miniconda3/etc/profile.d/conda.sh
conda activate search2

cd /data/sls/scratch/akolli/search_similarity

# --- Run server ---
python -m uvicorn search_endpoint:app --host 0.0.0.0 --port 8000
