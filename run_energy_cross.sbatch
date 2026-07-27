#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --partition=a100
#SBATCH --job-name=energy_cross
#SBATCH --output=logs/energy_cross_%j.out
#SBATCH --error=logs/energy_cross_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawspeech

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs tests/neutral_pitch
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

SEED=42
echo "node: $(hostname) | started: $(date) | seed: $SEED"

# ---------- Ensure duration flag exists ----------
python - <<'PYEOF'
f = 'drawspeech/dataset_plugin.py'
s = open(f).read()
if 'DRAWSPEECH_NO_DURATION' not in s:
    anchor = '    return {\n        "phoneme_idx": phoneme_idx,'
    inject = ('    import os as _os\n'
              '    if _os.environ.get("DRAWSPEECH_NO_DURATION") == "1":\n'
              '        duration = ""\n')
    assert anchor in s, "anchor not found"
    s = s.replace(anchor, inject + anchor, 1)
    open(f, 'w').write(s)
    print("patched: DRAWSPEECH_NO_DURATION added")
else:
    print("flag already present")
PYEOF

# ---------- 1. Energy cross-sketches ----------
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
cross_dir = 'data/dataset/metadata/ljspeech/energy_cross_sketches'
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
    src_e = f'data/dataset/metadata/ljspeech/phoneme_level/energy/LJSpeech-energy-{source}.npy'
    if not os.path.exists(src_e): continue
    src = np.load(src_e)
    w = min(5, len(src))
    if w % 2 == 0: w += 1
    sketch = savgol_filter(src, window_length=w, polyorder=2) if w >= 3 else src
    t_n = len(parts[2].strip('{}').split())
    sketch = np.interp(np.linspace(0,1,t_n), np.linspace(0,1,len(sketch)), sketch)
    np.save(f'{cross_dir}/{target}_energy_cross_sketch.npy', sketch)
    saved += 1
print(f"Energy cross-sketches saved: {saved}/{len(test_lines)}")
PYEOF

# ---------- 2. Build JSON (neutral pitch + cross energy) ----------
python <<PYEOF
import json, os, numpy as np

with open('data/dataset/metadata/ljspeech/test.json') as f:
    test_data = json.load(f)['data']

os.makedirs('tests/neutral_pitch', exist_ok=True)
data = []
for item in test_data:
    b = os.path.splitext(os.path.basename(item['wav']))[0]
    esk = f'data/dataset/metadata/ljspeech/energy_cross_sketches/{b}_energy_cross_sketch.npy'
    if not os.path.exists(esk): continue
    n_ph = len(item.get('phonemes','').strip('{}').split())
    if n_ph == 0: continue
    npath = f'tests/neutral_pitch/{b}_neutral_pitch.npy'
    np.save(npath, np.ones(n_ph) * 0.5)
    data.append({
        'wav': item['wav'],
        'transcription': item.get('transcription',''),
        'phonemes': item.get('phonemes',''),
        'pitch_sketch': npath,
        'energy_sketch': esk,
        'pitch': '', 'energy': '',
        'duration': f'data/dataset/metadata/ljspeech/phoneme_level/duration/LJSpeech-duration-{b}.npy',
    })

with open('tests/inference_energy_cross.json', 'w') as f:
    json.dump({'data': data}, f, indent=2)
print(f"Energy JSON samples: {len(data)}")
PYEOF

# ---------- 3a. Inference WITH duration ----------
echo "=========== ENERGY CROSS: DURATION PROVIDED ==========="
python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference tests/inference_energy_cross.json \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

DIR_WITH=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
echo "RMSE dir (with duration): $DIR_WITH"
python compute_rmse.py --generated_dir "$DIR_WITH" --original_dir data/dataset/LJSpeech-1.1/wavs

# ---------- 3b. Inference WITHOUT duration ----------
echo "=========== ENERGY CROSS: DURATION PREDICTED ==========="
DRAWSPEECH_NO_DURATION=1 python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
    --list_inference tests/inference_energy_cross.json \
    --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

DIR_WITHOUT=$(ls -dt log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* | head -1)
echo "RMSE dir (no duration): $DIR_WITHOUT"
python compute_rmse.py --generated_dir "$DIR_WITHOUT" --original_dir data/dataset/LJSpeech-1.1/wavs

echo "finished: $(date)"
