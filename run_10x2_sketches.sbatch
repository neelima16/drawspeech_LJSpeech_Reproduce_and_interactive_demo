#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --job-name=tenx2
#SBATCH --output=logs/tenx2_%j.out
#SBATCH --error=logs/tenx2_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -e
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawspeech
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs tests/tenx2
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

echo "node: $(hostname) | started: $(date)"

python build_tenx2.py

echo "===== VARIANT A ====="
python drawspeech/infer.py --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml --list_inference tests/tenx2/inference_A.json --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt
DIR_A=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
echo "DIR_A: $DIR_A"
echo "--- A vs ORIGINAL ---"
python compute_rmse.py --generated_dir "$DIR_A" --original_dir data/dataset/LJSpeech-1.1/wavs
echo "--- A vs SOURCE ---"
python compute_rmse_10_utterances.py --generated_dir "$DIR_A" --mapping tests/tenx2/map_A.json

echo "===== VARIANT B ====="
python drawspeech/infer.py --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml --list_inference tests/tenx2/inference_B.json --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt
DIR_B=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
echo "DIR_B: $DIR_B"
echo "--- B vs ORIGINAL ---"
python compute_rmse.py --generated_dir "$DIR_B" --original_dir data/dataset/LJSpeech-1.1/wavs
echo "--- B vs SOURCE ---"
python compute_rmse_10_utterances.py --generated_dir "$DIR_B" --mapping tests/tenx2/map_B.json

echo "finished: $(date)"
