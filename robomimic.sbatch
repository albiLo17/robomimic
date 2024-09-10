#!/usr/bin/env bash
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
## #SBATCH --constrain='balrog|belegost|gondor|khazadum|rivendell|shelob|shire|smaug'

source "/Midgard/home/longhini/miniconda3/etc/profile.d/conda.sh"
conda activate robomimi

python3 /Midgard/home/longhini/robomimic/robomimic/scripts/train_cluster.py --config "$CONFIG_FILE"