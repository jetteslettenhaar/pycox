#!/bin/bash
#SBATCH --ntasks=1     
#SBATCH --mem=14G      
#SBATCH --gres=gpu:1   
#SBATCH --exclude=gpu002
#SBATCH -p short       
#SBATCH -t 04:00:00
#SBATCH -o /trinity/home/r098372/pycox/output/out_%j.log
#SBATCH -e /trinity/home/r098372/pycox/output/error_%j.log

module purge
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

source /trinity/home/r098372/pycox/venv_new/bin/activate

python Model_jette_imaging.py
