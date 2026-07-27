#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=03:00:00
#SBATCH --job-name=pitch_cross_nodur
#SBATCH --output=logs/pitch_cross_nodur_%j.out
#SBATCH --error=logs/pitch_cross_nodur_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawspeech

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

echo "node: $(hostname) | started: $(date)"

# ---------- Ensure the DRAWSPEECH_NO_DURATION flag exists ----------
python - <<'PYEOF'
f = 'drawspeech/dataset_plugin.py'
s = open(f).read()
if 'DRAWSPEECH_NO_DURATION' not in s:
    anchor = '    return {\n        "phoneme_idx": phoneme_idx,'
    inject = (
        '    import os as _os\n'
        '    if _os.environ.get("DRAWSPEECH_NO_DURATION") == "1":\n'
        '        duration = ""\n'
    )
    assert anchor in s, "anchor not found — inspect dataset_plugin.py return dict"
    s = s.replace(anchor, inject + anchor, 1)
    open(f, 'w').write(s)
    print("patched: DRAWSPEECH_NO_DURATION flag added")
else:
    print("flag already present")
PYEOF

# ---------- Inference with duration OFF (reuse existing JSON/sketches) ----------
DRAWSPEECH_NO_DURATION=1 python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference tests/inference_pitch_cross.json \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

# ---------- RMSE ----------
LATEST_DIR=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
echo "RMSE dir: $LATEST_DIR"
python compute_rmse.py --generated_dir "$LATEST_DIR" --original_dir data/dataset/LJSpeech-1.1/wavs

echo "finished: $(date)"
