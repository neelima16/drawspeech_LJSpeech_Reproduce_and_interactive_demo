#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=03:00:00
#SBATCH --job-name=pitch_cross
#SBATCH --output=logs/pitch_cross_%j.out
#SBATCH --error=logs/pitch_cross_%j.err
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

SEED=42
echo "node: $(hostname) | started: $(date) | seed: $SEED"

# ---------- 1. Generate pitch cross-sketches (test set only) ----------
python <<PYEOF
import os, json, random, numpy as np
from scipy.signal import savgol_filter

with open('data/dataset/metadata/ljspeech/test.json') as f:
    test_items = json.load(f)['data']
test_basenames = [os.path.splitext(os.path.basename(i['wav']))[0] for i in test_items]

source_pool = []
for split in ['train.json', 'val.json']:
    with open(f'data/dataset/metadata/ljspeech/{split}') as f:
        for i in json.load(f)['data']:
            b = os.path.splitext(os.path.basename(i['wav']))[0]
            if b not in test_basenames:
                source_pool.append(b)

random.seed($SEED)
cross_dir = 'data/dataset/metadata/ljspeech/pitch_cross_sketches'
os.makedirs(cross_dir, exist_ok=True)

with open('data/dataset/metadata/ljspeech/phoneme_level/metadata.txt') as f:
    all_lines = [l.strip().split('|') for l in f if l.strip()]
test_lines = [l for l in all_lines if l[0] in test_basenames]

saved = 0
for parts in test_lines:
    target = parts[0]
    possible = [b for b in source_pool if b != target]
    if not possible: continue
    source = random.choice(possible)
    src_pitch = f'data/dataset/metadata/ljspeech/phoneme_level/pitch/LJSpeech-pitch-{source}.npy'
    if not os.path.exists(src_pitch): continue
    src = np.load(src_pitch)
    w = min(5, len(src))
    if w % 2 == 0: w += 1
    sketch = savgol_filter(src, window_length=w, polyorder=2) if w >= 3 else src
    t_n = len(parts[2].strip('{}').split())
    sketch = np.interp(np.linspace(0,1,t_n), np.linspace(0,1,len(sketch)), sketch)
    np.save(f'{cross_dir}/{target}_cross_sketch.npy', sketch)
    saved += 1
print(f"Pitch cross-sketches saved: {saved}/{len(test_lines)}")
PYEOF

# ---------- 2. Build inference JSON ----------
python <<PYEOF
import json, os
with open('data/dataset/metadata/ljspeech/test.json') as f:
    test_data = json.load(f)['data']
data = []
for item in test_data:
    b = os.path.splitext(os.path.basename(item['wav']))[0]
    sketch = f'data/dataset/metadata/ljspeech/pitch_cross_sketches/{b}_cross_sketch.npy'
    if not os.path.exists(sketch): continue
    data.append({
        'wav': item['wav'],
        'transcription': item.get('transcription',''),
        'phonemes': item.get('phonemes',''),
        'pitch_sketch': sketch,
        'energy_sketch': f'data/dataset/metadata/ljspeech/phoneme_level/energy/LJSpeech-energy-{b}.npy',
        'pitch': '', 'energy': '',
        'duration': f'data/dataset/metadata/ljspeech/phoneme_level/duration/LJSpeech-duration-{b}.npy',
    })
with open('tests/inference_pitch_cross.json', 'w') as f:
    json.dump({'data': data}, f, indent=2)
print(f"Pitch JSON samples: {len(data)}")
PYEOF

# ---------- 3. Inference ----------
python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference tests/inference_pitch_cross.json \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

# ---------- 4. RMSE ----------
LATEST_DIR=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
echo "RMSE dir: $LATEST_DIR"
python compute_rmse.py --generated_dir "$LATEST_DIR" --original_dir data/dataset/LJSpeech-1.1/wavs

echo "finished: $(date)"
