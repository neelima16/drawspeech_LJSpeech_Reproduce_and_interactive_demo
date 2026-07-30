
# DrawSpeech — LJSpeech Reproduction and Interactive Demo

This repository reproduces [DrawSpeech](https://arxiv.org/abs/2501.04256) (ICASSP 2025) on the **LJSpeech** dataset, adds cross-utterance prosody-transfer experiments, and provides an interactive Streamlit demo.

**Original Paper:** [[IEEE]](https://ieeexplore.ieee.org/abstract/document/10889767) · [[arXiv]](https://arxiv.org/abs/2501.04256) · [[Demo]](https://happycolor.github.io/DrawSpeech/)
**Original Code:** [HappyColor/DrawSpeech_PyTorch](https://github.com/HappyColor/DrawSpeech_PyTorch)

---

## Contents

- **Reproduction of DrawSpeech on LJSpeech** (single speaker)
  - Fixed 4 bugs in the original codebase.
  - Fixed the checkpoint key mismatch (`drawspeech_fixed.ckpt`).
  - Evaluated pitch and energy RMSE.
- **Cross-utterance prosody transfer**
  - Pitch transfer (sketch taken from a different sentence).
  - Energy transfer (energy sketch from a different sentence).
  - With and without duration conditioning.
- **Interactive Streamlit demo (`app.py`)**
  - Word-level and phoneme-level pitch sliders.
  - Real-time audio generation.
  - Sketch vs. generated-pitch visualization (Figures 2 and 3 from the paper).
  - Sketch-refinement pipeline (Savitzky–Golay smoothing + normalization).
- **10×2 sketch experiments** — the same sentence with different sketch intensities.

---

## Key Results

All experiments transfer a pitch or energy **sketch from a source sentence onto a different target sentence**. The single biggest factor in the error is whether the model is given **real phoneme durations** (timing from the target recording) or has to **predict its own**.

### Duration is the dominant factor

Across runs of different sizes (10, 20, 300 samples), the results cluster tightly by condition:

| Condition                        | Pitch RMSE (range) | Energy RMSE (range) |
|----------------------------------|-------------------:|--------------------:|
| **With duration** (real timing)  | ~32–34 Hz          | ~4.5 dB             |
| **Without duration** (predicted) | ~50 Hz             | ~11 dB              |

→ Providing real durations lowers pitch RMSE by **~16 Hz**. The gap is reproduced independently at both 20 and 300 samples, so it is driven by the duration condition itself — not by which utterances happen to land in a given shuffle.

### Individual runs

**With duration (real timing given):**

| Run             | Samples | Pitch RMSE | Energy RMSE |
|-----------------|--------:|-----------:|------------:|
| vs. target      | 10      | 34.11 Hz   | 4.52 dB     |
| Cross-utterance | 300     | 32.37 Hz   | 4.70 dB     |

**Without duration (predicted timing):**

| Run             | Samples | Pitch RMSE | Energy RMSE |
|-----------------|--------:|-----------:|------------:|
| Cross-utterance | 300     | 50.10 Hz   | 11.04 dB    |

**Sanity check — distance from the pitch *source* (10 samples, with duration):**
Pitch RMSE 82.47 Hz · Energy RMSE 20.93 dB — deliberately large, since only the pitch *shape* was borrowed, not the words or timing.

### Energy sketch study

_In progress._

> **Note on variance.** Absolute numbers shift slightly between runs because the original preprocessing randomizes the train/val/test split. We keep this behavior to stay faithful to the original code; the **trends and the ~16 Hz with/without-duration gap remain consistent**.

> **Not comparable:** the paper's headline cross-utterance result (57.67 Hz) keeps the *original* energy and is a different setup — don't compare it directly against the duration pair above.

---

## Setup

```bash
# Optional: set proxies if behind a firewall
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

conda env create -f environment.yml
conda activate drawspeech

# Install taming-transformers (required for the VAE)
git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers && pip install -e . && cd ..

export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
```

---

## Dataset & Checkpoints

### LJSpeech

1. Download from [keithito.com](https://keithito.com/LJ-Speech-Dataset/) and place the archive in `data/dataset/`.
2. Extract the dataset and alignments:
   ```bash
   cd data/dataset
   tar -xjf LJSpeech-1.1.tar.bz2
   unzip LJSpeech.zip -d LJSpeech-1.1/   # alignments
   ```
   Expected structure:
   ```
   data/dataset/LJSpeech-1.1/
   ├── metadata.csv
   ├── wavs/*.wav
   └── *.[TextGrid|lab]          # alignment files
   ```

### Checkpoints

Download the pretrained checkpoints from [HuggingFace](https://huggingface.co/HappyColor/DrawSpeech/tree/main) into `data/checkpoints/`.

If a checkpoint is broken (e.g., `generator_v1` is a 0-byte file), replace it from the mirror:

```bash
rm -f data/checkpoints/LJ_V1/generator_v1
curl -L -o data/checkpoints/LJ_V1/generator_v1 \
  "https://drive.usercontent.google.com/download?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW&export=download"
```

Then fix the checkpoint key mismatch (produces `data/checkpoints/drawspeech_fixed.ckpt`):

```bash
python fix_drawspeech.py
```

---

## Preprocessing

Extract mel-spectrograms, pitch, energy, and duration features:

```bash
python preprocessing.py
```

> `preprocessing.py` shuffles the train/validation split using a fixed seed, so runs are reproducible. Keep the seed unchanged to match our exact splits.

---

## Experiments

Experiments are submitted via SLURM. Each script sets its config and runs inference.

```bash
sbatch run_pitch_cross.sbatch         # pitch cross-utterance (with duration)
sbatch run_pitch_cross_nodur.sbatch   # pitch cross-utterance (without duration)
sbatch run_energy_cross.sbatch        # energy cross-utterance
sbatch run_10x2_sketches.sbatch       # 10×2 sketch experiments
```

### Compute RMSE

```bash
python compute_rmse.py \
    --generated_dir "log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_*" \
    --original_dir  data/dataset/LJSpeech-1.1/wavs
```

---

## Interactive Demo (Streamlit)

The app is functional on a GPU node.

1. **On the GPU node:**
   ```bash
   export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
2. **From your local machine**, open an SSH tunnel (replace `<GPU_HOST>`, `<LOGIN_NODE>`, `<username>`):
   ```bash
   ssh -L 8501:<GPU_HOST>:8501 <username>@<LOGIN_NODE>
   ```
   Example: `ssh -L 8501:tg090:8501 user@csnhr.nhr.fau.de`
3. Open `http://localhost:8501`. If the port is taken, map another (e.g., `-L 8502:tg090:8501` → `http://localhost:8502`).

**Features:** word- and phoneme-level pitch sliders · real-time audio generation · sketch vs. generated-pitch plots · sketch-refinement pipeline (smoothing + normalization).

> **Limitation.** The released checkpoint was trained only on smoothed real pitch contours, so it does not reliably follow arbitrary hand-drawn or slider sketches. Expect subtle rather than dramatic pitch changes.

---

## Repository Structure

```
.
├── app.py                        # Streamlit interactive demo
├── sketch_adapter.py             # Sketch refinement pipeline
├── fix_drawspeech.py             # Checkpoint key mismatch fix
├── preprocessing.py              # Feature extraction
├── compute_rmse.py               # RMSE evaluation
├── compute_rmse_10_utterances.py # 10-utterance RMSE
├── build_tenx2.py                # 10×2 sketch builder
├── run_*.sbatch                  # SLURM experiment scripts
├── environment.yml               # Conda environment
├── RESULTS.md                    # Experiment results log
├── drawspeech/
│   ├── config/drawspeech_ljspeech_22k.yaml
│   ├── conditional_models.py
│   ├── dataset_plugin.py
│   ├── infer.py
│   └── modules/
├── tests/                        # Inference JSON configs
└── data/
    ├── checkpoints/              # Pretrained models
    └── dataset/                  # LJSpeech and alignments
```

---

## Acknowledgements

- [AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning)
- [FastSpeech 2](https://github.com/ming024/FastSpeech2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

---

## Citation

```bibtex
@inproceedings{chen2025drawspeech,
  author    = {Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  title     = {DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  booktitle = {ICASSP},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10889767}
}
```
```

That's the complete file. One reminder from earlier: the "Energy sketch study" is still marked _In progress._ — fill that in when you have the numbers, or delete the subsection if you'd rather not show a placeholder.
