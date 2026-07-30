
# DrawSpeech — LJSpeech Reproduction and Interactive Demo

This repository contains a reproduction of [DrawSpeech](https://arxiv.org/abs/2501.04256) (ICASSP 2025) on the **LJSpeech** dataset, along with cross-utterance prosody transfer experiments and an interactive Streamlit demo.

**Original Paper:** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10889767) [[arXiv]](https://arxiv.org/abs/2501.04256) [[Demo]](https://happycolor.github.io/DrawSpeech/)
**Original Code:** [HappyColor/DrawSpeech_PyTorch](https://github.com/HappyColor/DrawSpeech_PyTorch)

---

## What This Repository Contains

```
1. Reproduction of DrawSpeech on LJSpeech (single speaker)
   - Fixed 4 bugs in original codebase
   - Fixed checkpoint key mismatch (drawspeech_fixed.ckpt)
   - Evaluated pitch and energy RMSE

2. Cross-utterance prosody transfer experiments
   - Pitch cross-utterance (sketch from different sentence)
   - Energy cross-utterance (energy sketch from different sentence)
   - With and without duration conditioning

3. Interactive Streamlit Demo (app.py)
   - Word-level pitch sliders
   - Phoneme-level pitch sliders
   - Real-time audio generation
   - Sketch vs generated pitch visualization
   - Sketch refinement pipeline

4. 10x2 sketch experiments
   - Same sentence with different sketch intensities
```

---

## Key Results

### Pitch Cross-Utterance (LJSpeech test set, 300 utterances)

```
Pitch RMSE:   34.18 Hz
Energy RMSE:   4.54 dB
```

See `RESULTS.md` for full experiment log.

---

## Bug Fixes in Original Code

```
1. stft.py: pad_center keyword argument (librosa API change)
2. fastspeech2/modules.py: missing mel_mask computation
3. ddpm.py: undefined CLAP reference, strict=False loading
4. infer.py: handles both checkpoint formats
```

---

## Checkpoint Key Mismatch Fix

The released `drawspeech.ckpt` has old layer names. Fix with:

```bash
python fix_drawspeech.py
```

This creates `data/checkpoints/drawspeech_fixed.ckpt`.

---

## Environment Setup

```bash

#if needed, enable 
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

conda env create -f environment.yml
conda activate drawspeech

git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ..
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH


```

---

## Dataset and Checkpoints

### LJSpeech
Download from https://keithito.com/LJ-Speech-Dataset/ and place at:
```
data/dataset/LJSpeech-1.1/
├── metadata.csv
└── wavs/
```

### Alignments
Download from [Google Drive](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4) and unzip into `data/dataset/LJSpeech-1.1/`.

### Checkpoints
Download from [HuggingFace](https://huggingface.co/HappyColor/DrawSpeech/tree/main):

Place the checkpoints into data/checkpoints/

# 1. Extract the main LJSpeech dataset
tar -xjf LJSpeech-1.1.tar.bz2

# 2. Extract the alignments (LJSpeech.zip) into the LJSpeech-1.1 folder
unzip LJSpeech.zip -d LJSpeech-1.1/


Then run:
```bash
python fix_drawspeech.py
```

### Preprocessing

```bash
python preprocessing.py
```

---

## Experiments

### Pitch Cross-Utterance (with duration)

```bash
sbatch run_pitch_cross.sbatch
```

### Pitch Cross-Utterance (without duration)

```bash
sbatch run_pitch_cross_nodur.sbatch
```

### Energy Cross-Utterance

```bash
sbatch run_energy_cross.sbatch
```

### 10x2 Sketch Experiments

```bash
sbatch run_10x2_sketches.sbatch
```
Sometimes, the chcekpoint can be broken, 
use this to fix 
# Remove the broken file
rm -f ~/repro1/drawspeech_LJSpeech_Reproduce_and_interactive_demo/data/checkpoints/LJ_V1/generator_v1

# Download from the reliable mirror (Google Drive)
curl -L -o ~/repro1/drawspeech_LJSpeech_Reproduce_and_interactive_demo/data/checkpoints/LJ_V1/generator_v1 \
  "https://drive.usercontent.google.com/download?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW&export=download"
---

## Compute RMSE

```bash
python compute_rmse.py \
    --generated_dir log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* \
    --original_dir data/dataset/LJSpeech-1.1/wavs
```

---


## Interactive demo (work in progress)

An interactive Streamlit app lets you **draw pitch sketches** or use **word/phoneme sliders** and hear the result in real time. The app is still being polished, but you can test it on a GPU node:

```bash
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then, from your **local machine**, create an SSH tunnel. Replace `<GPU_HOST>` with the actual hostname of the GPU node (e.g., `tg090`), and `<LOGIN_NODE>` with your cluster's login address (e.g., `tinyx` or `csnhr.nhr.fau.de`):

```bash
ssh -L 8501:<GPU_HOST>:8501 <username>@<LOGIN_NODE>
```

For example, if your username is `iwi5408h` and the GPU node is `tg090`:

```bash
ssh -L 8501:tg090:8501 iwi5408h@csnhr.nhr.fau.de
```

Now open your browser and go to **http://localhost:8501**.  
(If port 8501 is blocked locally, use `-L 8502:...` and open `http://localhost:8502`.)

---


For a Mac user
Start the Streamlit app on the GPU node (as shown in your README).

Open Terminal on your Mac.

Create the SSH tunnel (replace tg090 with the actual GPU hostname and csnhr.nhr.fau.de with your login node):
```
bash
ssh -L 8501:tg090:8501 iwi5408h@csnhr.nhr.fau.de
```
If port 8501 is already in use on your Mac, use a different local port, e.g.:
```
bash
ssh -L 8502:tg090:8501 iwi5408h@csnhr.nhr.fau.de
```
Keep the Terminal window open.

Open your browser and go to http://localhost:8501 (or http://localhost:8502 if you used the alternate port).

Features:
- Word-level and phoneme-level pitch sliders
- Real-time audio generation
- Sketch vs generated pitch visualization (Figures 2 and 3 from paper)
- Sketch refinement pipeline (Savitzky-Golay smoothing + normalization)

---

## Repository Structure

```
DrawSpeech_PyTorch/
├── app.py                        # Streamlit interactive demo
├── sketch_adapter.py             # Sketch refinement pipeline
├── fix_drawspeech.py             # Checkpoint key mismatch fix
├── preprocessing.py              # Feature extraction
├── compute_rmse.py               # RMSE evaluation
├── compute_rmse_10_utterances.py # 10-utterance RMSE
├── build_tenx2.py                # 10x2 sketch builder
├── run_pitch_cross.sbatch        # Pitch cross experiment
├── run_pitch_cross_nodur.sbatch  # Pitch cross (no duration)
├── run_energy_cross.sbatch       # Energy cross experiment
├── run_10x2_sketches.sbatch      # 10x2 sketch experiment
├── environment.yml               # Conda environment
├── RESULTS.md                    # Experiment results log
├── drawspeech/
│   ├── config/
│   │   └── drawspeech_ljspeech_22k.yaml
│   ├── conditional_models.py     # Core model
│   ├── dataset_plugin.py         # Data loading
│   ├── infer.py                  # Inference script
│   └── modules/                  # Model components
├── tests/
│   ├── inference.json            # Basic inference test
│   ├── inference_pitch_cross.json
│   └── inference_energy_cross.json
└── data/
    └── dataset/
        └── metadata/
            └── ljspeech/
                ├── train.json
                ├── val.json
                └── test.json
```

---

## Monte Carlo Sampling

DrawSpeech uses DDIM sampling (200 steps) during inference. Outputs are stochastic — use fixed seed for reproducibility:

```python
from pytorch_lightning import seed_everything
seed_everything(0)
```

---

## Acknowledgements

* [AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning)
* [FastSpeech 2](https://github.com/ming024/FastSpeech2)
* [HiFi-GAN](https://github.com/jik876/hifi-gan)

---

## Citation

```bibtex
@INPROCEEDINGS{10889767,
  author={Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  booktitle={ICASSP 2025},
  title={DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  year={2025},
  doi={10.1109/ICASSP49660.2025.10889767}
}
```

