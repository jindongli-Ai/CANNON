#!/bin/bash
#SBATCH --job-name=CANNON
#SBATCH --output=CANNON-Output.log       
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
echo "Hello World! This is my CANNON job on Slurm."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
conda init
conda activate CANNON
cd CANNON
cd src
bash script/cora.sh
echo "Job completed successfully."