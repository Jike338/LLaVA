#!/bin/bash

#SBATCH --account=kpsounis_171
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=slurm_out/%x_%j.out


module load python
module load cuda

# Create and activate virtual environment
VENV_DIR="/scratch1/jikezhon/LLaVA/venv"
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install necessary Python packages inside the virtual environment
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Change directory to the working directory
cd /scratch1/jikezhon/LLaVA

# Run your Python script
python3 test_pytorch.py

# Deactivate the virtual environment after the job is done
deactivate
