# DrawSpeech — LJSpeech Reproduction and Interactive Demo

This repository reproduces [DrawSpeech](https://arxiv.org/abs/2501.04256) (ICASSP 2025) on the **LJSpeech** dataset, adds cross-utterance prosody-transfer experiments, and provides an interactive Streamlit demo.

**Original Paper:** [[IEEE]](https://ieeexplore.ieee.org/abstract/document/10889767) · [[arXiv]](https://arxiv.org/abs/2501.04256) · [[Demo]](https://happycolor.github.io/DrawSpeech/)  
**Original Code:** [HappyColor/DrawSpeech_PyTorch](https://github.com/HappyColor/DrawSpeech_PyTorch)

---

# Contents

- **Reproduction of DrawSpeech on LJSpeech** (single speaker)
  - Fixed four bugs in the original implementation.
  - Fixed the pretrained checkpoint key mismatch (`drawspeech_fixed.ckpt`).
  - Evaluated pitch and energy RMSE.
- **Cross-utterance prosody transfer**
  - Pitch transfer from a source utterance to a different target utterance.
  - Energy transfer from a source utterance to a different target utterance.
  - Experiments with and without duration conditioning.
- **Interactive Streamlit demo**
  - Word-level and phoneme-level pitch sliders.
  - Real-time speech synthesis.
  - Sketch vs. generated pitch visualization.
  - Sketch refinement (Savitzky–Golay smoothing + normalization).
- **10×2 sketch experiments**

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
|------|--------:|-----------:|------------:|
| vs. target | 10 | 34.11 Hz | 4.52 dB |
| Cross-utterance | 300 | 32.37 Hz | 4.70 dB |

#### Without duration

| Run | Samples | Pitch RMSE | Energy RMSE |
|------|--------:|-----------:|------------:|
| Cross-utterance | 300 | 50.10 Hz | 11.04 dB |

### Sanity Check

Distance from the **source** utterance (10 samples, with duration):

- Pitch RMSE: **82.47 Hz**
- Energy RMSE: **20.93 dB**

The error is intentionally large because only the pitch contour is transferred—not the linguistic content or timing.

### Energy Transfer

| Run | Samples | Pitch RMSE | Energy RMSE |
|------|--------:|-----------:|------------:|
| With duration | 300 | 32.37 Hz | 4.71 dB |
| Without duration | 300 | 48.19 Hz | 10.93 dB |

Providing real durations reduces pitch RMSE by roughly **16 Hz** and cuts energy RMSE by more than half.

### Main Takeaway

The absolute RMSE varies slightly because preprocessing randomly generates the train/validation split (using a fixed seed), but the trend is consistent:

| Condition | Pitch RMSE |
|-----------|-----------:|
| With duration | ~32–34 Hz |
| Without duration | ~48–51 Hz |
| Improvement | ~14–16 Hz |

---

# Setup

## 1. Clone the repository

```bash
git clone <repository-url>
cd DrawSpeech_LibriTTS
```

## 2. Create the conda environment

```bash
# Optional: set proxies if behind a firewall
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

conda env create -f environment.yml
conda activate drawspeech
```

## 3. Download pretrained checkpoints

Create the checkpoint directory.

```bash
mkdir -p data/checkpoints/LJ_V1
```

Install the Hugging Face client.

```bash
pip install huggingface_hub
```

Download the DrawSpeech checkpoints.

```bash
python -c "
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='HappyColor/DrawSpeech',
    filename='vae.ckpt',
    local_dir='data/checkpoints'
)

hf_hub_download(
    repo_id='HappyColor/DrawSpeech',
    filename='drawspeech.ckpt',
    local_dir='data/checkpoints'
)
"
```

Download the HiFi-GAN vocoder.

```bash
curl -L -o data/checkpoints/LJ_V1/config.json \
https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json

curl -L -o data/checkpoints/LJ_V1/generator_v1 \
"https://drive.usercontent.google.com/download?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW&export=download"
```

## 4. Install taming-transformers

```bash
git clone https://github.com/CompVis/taming-transformers.git

cd taming-transformers
pip install -e .
cd ..
```

## 5. Download the LPIPS VGG model

```bash
mkdir -p taming/modules/autoencoder/lpips

curl -L \
-o taming/modules/autoencoder/lpips/vgg.pth \
"https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
```

## 6. Fix the checkpoint

The released checkpoint uses different parameter names than expected by the inference code.

Run:

```bash
python fix_drawspeech.py
```

This creates

```
data/checkpoints/drawspeech_fixed.ckpt
```

## 7. Set the Python path

```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
```

---

# Dataset

Download the **LJSpeech** dataset from

https://keithito.com/LJ-Speech-Dataset/

Place the archive inside

```
data/dataset/
```

Extract the dataset and alignments.

```bash
cd data/dataset

tar -xjf LJSpeech-1.1.tar.bz2
unzip LJSpeech.zip -d LJSpeech-1.1/
```

Expected directory structure:

```
data/dataset/LJSpeech-1.1/
├── metadata.csv
├── wavs/
├── *.lab
└── *.TextGrid
```

---

# Preprocessing

Extract mel spectrograms, pitch, energy and duration features.

```bash
python preprocessing.py
```

The preprocessing script uses a fixed random seed to create train and validation splits, making experiments reproducible.

---

# Experiments

Experiments are executed through SLURM.

### Pitch transfer (with duration)

```bash
sbatch run_pitch_cross.sbatch
```

### Pitch transfer (without duration)

```bash
sbatch run_pitch_cross_nodur.sbatch
```

### Energy transfer

```bash
sbatch run_energy_cross.sbatch
```

### 10×2 sketch experiment

```bash
sbatch run_10x2_sketches.sbatch
```

---

# RMSE Evaluation

Compute pitch and energy RMSE between generated and reference speech.

```bash
python compute_rmse.py \
    --generated_dir log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* \
    --original_dir data/dataset/LJSpeech-1.1/wavs
```

---

# Interactive Demo (Streamlit)

Launch the demo on a GPU node.

```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH

streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0
```

From your local machine, create an SSH tunnel.

```bash
ssh -L 8501:<GPU_HOST>:8501 <username>@<LOGIN_NODE>
```

Example

```bash
ssh -L 8501:tg090:8501 user@csnhr.nhr.fau.de
```

Open

```
http://localhost:8501
```

### Features

- Word-level pitch editing
- Phoneme-level pitch editing
- Real-time speech synthesis
- Sketch vs. generated pitch visualization
- Sketch refinement pipeline
- Audio playback

### Limitation

The released checkpoint was trained using smoothed real pitch contours only. Consequently, manually drawn sketches or slider edits produce subtle changes rather than perfectly following arbitrary user input.

---

# Repository Structure

```text
.
├── app.py
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
│   ├── config/
│   ├── modules/
│   ├── infer.py
│   ├── dataset_plugin.py
│   └── conditional_models.py
├── tests/
└── data/
    ├── checkpoints/
    └── dataset/
```

---

# Acknowledgements

This work builds upon the following open-source projects:

- DrawSpeech
- AudioLDM
- FastSpeech2
- HiFi-GAN
- taming-transformers

---

# Citation

If you use this repository, please cite the original DrawSpeech paper.

```bibtex
@inproceedings{chen2025drawspeech,
  author    = {Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  title     = {DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  booktitle = {ICASSP},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10889767}
}
```
