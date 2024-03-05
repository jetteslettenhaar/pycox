#!/bin/bash
#SBATCH --ntasks=1     
#SBATCH --mem=14G      
#SBATCH --gres=gpu:1   
#SBATCH -p short       
#SBATCH -t 01:00:00
#SBATCH -o /trinity/home/r098372/pycox/output/out_%j.log
#SBATCH -e /trinity/home/r098372/pycox/output/error_%j.log

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load CUDA/10.1.243-GCC-8.3.0

source /trinity/home/r098372/pycox/venv/bin/activate

python Model_jette.py
