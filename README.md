# DrawSpeech — LJSpeech Reproduction and Interactive Demo

This repository reproduces [DrawSpeech](https://arxiv.org/abs/2501.04256) (ICASSP 2025) on the **LJSpeech** dataset, adds cross-utterance prosody-transfer experiments, and provides an interactive Streamlit demo.

**Original Paper:** [[IEEE]](https://ieeexplore.ieee.org/abstract/document/10889767) · [[arXiv]](https://arxiv.org/abs/2501.04256) · [[Demo]](https://happycolor.github.io/DrawSpeech/)  
**Original Code:** [HappyColor/DrawSpeech_PyTorch](https://github.com/HappyColor/DrawSpeech_PyTorch)

---

# Contents

- Reproduction of DrawSpeech on LJSpeech (single speaker)
  - Fixed four bugs in the original implementation.
  - Fixed the pretrained checkpoint key mismatch (`drawspeech_fixed.ckpt`).
  - Evaluated pitch and energy RMSE.
- Cross-utterance prosody transfer
  - Pitch transfer between different utterances.
  - Energy transfer between different utterances.
  - Experiments with and without duration conditioning.
- Interactive Streamlit demo
  - Word-level and phoneme-level pitch sliders.
  - Real-time speech synthesis.
  - Sketch vs. generated pitch visualization.
  - Sketch refinement (Savitzky–Golay smoothing + normalization).
- 10×2 sketch experiments.

---

# Key Results

All experiments transfer a pitch or energy **sketch from a source sentence onto a different target sentence**. The largest factor affecting performance is whether the model is given **ground-truth phoneme durations** or must predict them.

## Duration is the dominant factor

| Condition | Pitch RMSE | Energy RMSE |
|-----------|-----------:|------------:|
| **With duration** | ~32–34 Hz | ~4.5 dB |
| **Without duration** | ~48–51 Hz | ~11 dB |

Providing real durations consistently lowers pitch RMSE by approximately **14–16 Hz**.

### Individual Runs

#### With duration

| Run | Samples | Pitch RMSE | Energy RMSE |
|-----|--------:|-----------:|------------:|
| vs. target | 10 | 34.11 Hz | 4.52 dB |
| Cross-utterance | 300 | 32.37 Hz | 4.70 dB |

#### Without duration

| Run | Samples | Pitch RMSE | Energy RMSE |
|-----|--------:|-----------:|------------:|
| Cross-utterance | 300 | 50.10 Hz | 11.04 dB |

### Sanity Check

Comparing generated speech against the **source** utterance (instead of the target):

- Pitch RMSE: **82.47 Hz**
- Energy RMSE: **20.93 dB**

The error is intentionally large because only the pitch contour is transferred—not the words or timing.

### Energy Transfer

| Run | Samples | Pitch RMSE | Energy RMSE |
|-----|--------:|-----------:|------------:|
| With duration | 300 | 32.37 Hz | 4.71 dB |
| Without duration | 300 | 48.19 Hz | 10.93 dB |

Providing real durations reduces pitch RMSE by roughly **16 Hz** and cuts energy RMSE by more than half.

### Main Takeaway

The exact RMSE changes slightly because preprocessing randomly generates the train/validation split (using a fixed seed), but the trend is stable:

| Condition | Pitch RMSE |
|-----------|-----------:|
| With duration | ~32–34 Hz |
| Without duration | ~48–51 Hz |
| Improvement | ~14–16 Hz |

---

# Setup

## 1. Clone the repository

```bash
git clone <your_repository_url>
cd DrawSpeech_LibriTTS
```

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate drawspeech
```

If you are working behind a proxy:

```bash
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

---

## 3. Automatic Setup (Recommended)

Run

```bash
chmod +x setup.sh
./setup.sh
```

The setup script automatically

- downloads the DrawSpeech checkpoints from Hugging Face,
- downloads the HiFi-GAN vocoder,
- clones and installs `taming-transformers`,
- downloads the LPIPS VGG model,
- creates the required directory structure.

Afterwards, fix the checkpoint naming:

```bash
python fix_drawspeech.py
```

Finally, expose the repository to Python:

```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
```

---

# Dataset

Download **LJSpeech** from

https://keithito.com/LJ-Speech-Dataset/

Place the archive inside

```
data/dataset/
```

Then extract

```bash
cd data/dataset

tar -xjf LJSpeech-1.1.tar.bz2
unzip LJSpeech.zip -d LJSpeech-1.1/
```

Expected directory:

```
data/dataset/LJSpeech-1.1/
├── metadata.csv
├── wavs/
├── *.lab
└── *.TextGrid
```

---

# Preprocessing

Extract mel spectrograms, pitch, duration and energy features.

```bash
python preprocessing.py
```

---

# Running Experiments

Experiments are executed through SLURM.

```bash
sbatch run_pitch_cross.sbatch
```

Pitch transfer without duration

```bash
sbatch run_pitch_cross_nodur.sbatch
```

Energy transfer

```bash
sbatch run_energy_cross.sbatch
```

10×2 sketch experiment

```bash
sbatch run_10x2_sketches.sbatch
```

---

# RMSE Evaluation

```bash
python compute_rmse.py \
    --generated_dir log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* \
    --original_dir data/dataset/LJSpeech-1.1/wavs
```

---

# Streamlit Demo

Start the application on the GPU node.

```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0
```

Create an SSH tunnel from your local machine.

```bash
ssh -L 8501:<GPU_HOST>:8501 <username>@<LOGIN_NODE>
```

Example

```bash
ssh -L 8501:tg090:8501 user@csnhr.nhr.fau.de
```

Then open

```
http://localhost:8501
```

### Features

- Word-level pitch editing
- Phoneme-level pitch editing
- Real-time speech synthesis
- Pitch visualization
- Sketch refinement
- Audio playback

### Current Limitation

The released checkpoint was trained only using smoothed real pitch contours. It therefore follows manually drawn sketches only approximately, rather than exactly.

---

# Repository Structure

```
.
├── app.py
├── setup.sh
├── sketch_adapter.py
├── fix_drawspeech.py
├── preprocessing.py
├── compute_rmse.py
├── compute_rmse_10_utterances.py
├── build_tenx2.py
├── run_*.sbatch
├── environment.yml
├── RESULTS.md
├── drawspeech/
├── tests/
└── data/
    ├── checkpoints/
    └── dataset/
```

---

# Acknowledgements

- AudioLDM
- FastSpeech2
- HiFi-GAN
- DrawSpeech

---

# Citation

```bibtex
@inproceedings{chen2025drawspeech,
  author    = {Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  title     = {DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  booktitle = {ICASSP},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10889767}
}
```
