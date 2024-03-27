#!/bin/bash
#SBATCH --mail-user=nikidorny@icc.ru
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=79G
#SBATCH --time=7-07:00:00
#SBATCH --job-name=rutabert-local

source venv/bin/activate &&\
    pip install -r requirements.txt &&\
    python3 train.py 2> logs/error_train.log &&\
    python3 test.py 2> logs/error_test.log
