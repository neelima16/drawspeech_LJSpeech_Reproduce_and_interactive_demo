
# DrawSpeech — LJSpeech Reproduction and Interactive Demo

This repository reproduces [DrawSpeech](https://arxiv.org/abs/2501.04256) (ICASSP 2025) on the **LJSpeech** dataset, adds cross-utterance prosody transfer experiments, and provides an interactive Streamlit demo.

**Original Paper:** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10889767) [[arXiv]](https://arxiv.org/abs/2501.04256) [[Demo]](https://happycolor.github.io/DrawSpeech/)  
**Original Code:** [HappyColor/DrawSpeech_PyTorch](https://github.com/HappyColor/DrawSpeech_PyTorch)

---

## What This Repository Contains

- **Reproduction of DrawSpeech on LJSpeech** (single speaker)  
  - Fixed 4 bugs in the original codebase.  
  - Fixed checkpoint key mismatch (`drawspeech_fixed.ckpt`).  
  - Evaluated pitch and energy RMSE.

- **Cross‑utterance prosody transfer experiments**  
  - Pitch transfer (sketch from a different sentence).  
  - Energy transfer (energy sketch from a different sentence).  
  - With and without duration conditioning.

- **Interactive Streamlit demo (`app.py`)**  
  - Word‑level and phoneme‑level pitch sliders.  
  - Real‑time audio generation.  
  - Sketch vs. generated pitch visualization (Figures 2 and 3 from the paper).  
  - Sketch refinement pipeline (Savitzky‑Golay smoothing + normalisation).

- **10×2 sketch experiments** – same sentence with different sketch intensities.

---

## Key Results

| Metric | Value |
|--------|-------|
| Pitch RMSE (cross‑utterance) | 34.18 Hz |
| Energy RMSE (cross‑utterance) | 4.54 dB |

See [`RESULTS.md`](RESULTS.md) for the full experiment log.

---

## Setup

### Environment

```bash
# Optional: set proxies if behind a firewall
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

conda env create -f environment.yml
conda activate drawspeech

# Install taming‑transformers (required for VAE)
git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ..
export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
```

---

## Dataset & Checkpoints

### LJSpeech Dataset

1. Download from [keithito.com](https://keithito.com/LJ-Speech-Dataset/) and place the archive in `data/dataset/`.
2. Extract the main dataset and alignments:
   ```bash
   cd data/dataset
   tar -xjf LJSpeech-1.1.tar.bz2
   unzip LJSpeech.zip -d LJSpeech-1.1/   # alignments
   ```
   The final structure should be:
   ```
   data/dataset/LJSpeech-1.1/
   ├── metadata.csv
   ├── wavs/
   │   └── *.wav
   └── *.[TextGrid|lab]          # alignment files
   ```

### Checkpoints

Download the pretrained checkpoints from [HuggingFace](https://huggingface.co/HappyColor/DrawSpeech/tree/main) and place them in `data/checkpoints/`.

**If the downloaded checkpoint is broken** (e.g., `generator_v1` is a 0‑byte file), remove it and fetch a working copy from the mirror:

```bash
rm -f data/checkpoints/LJ_V1/generator_v1
curl -L -o data/checkpoints/LJ_V1/generator_v1 \
  "https://drive.usercontent.google.com/download?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW&export=download"
```

Then fix the checkpoint key mismatch:

```bash
python fix_drawspeech.py
```

This creates `data/checkpoints/drawspeech_fixed.ckpt`.

---

## Preprocessing

Extract features (mel‑spectrograms, pitch, energy, duration) with:

```bash
python preprocessing.py
```

> **Note:** `preprocessing.py` shuffles the dataset for training/validation split. The shuffle uses a fixed random seed, so results are reproducible. If you want the exact same split as used in our experiments, ensure you do not change the seed.

---

## Experiments

All experiments are submitted via SLURM scripts. Each script sets the appropriate config and runs inference.

### Pitch Cross‑Utterance (with duration)
```bash
sbatch run_pitch_cross.sbatch
```

### Pitch Cross‑Utterance (without duration)
```bash
sbatch run_pitch_cross_nodur.sbatch
```

### Energy Cross‑Utterance
```bash
sbatch run_energy_cross.sbatch
```

### 10×2 Sketch Experiments
```bash
sbatch run_10x2_sketches.sbatch
```

---

## Compute RMSE

After generating audio, compute RMSE against the original LJSpeech wavs:

```bash
python compute_rmse.py \
    --generated_dir log/latent_diffusion/config/drawspeech_ljspeech_22k/infer_* \
    --original_dir data/dataset/LJSpeech-1.1/wavs
```

---

## Interactive Demo (Streamlit)

The Streamlit app is still being polished, but it is functional on a GPU node.

1. **On the GPU node**, run:
   ```bash
   export PYTHONPATH=$(pwd):$(pwd)/taming-transformers:$PYTHONPATH
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

2. **From your local machine**, create an SSH tunnel. Replace `<GPU_HOST>` with the actual hostname (e.g., `tg090`) and `<LOGIN_NODE>` with your cluster's login address (e.g., `csnhr.nhr.fau.de`):
   ```bash
   ssh -L 8501:<GPU_HOST>:8501 <username>@<LOGIN_NODE>
   ```
   

3. Open your browser at `http://localhost:8501` (or use a different local port if 8501 is occupied, e.g., `-L 8502:tgxxx:8501` and visit `http://localhost:8502`).

**Features:**
- Word‑level and phoneme‑level pitch sliders.
- Real‑time audio generation.
- Sketch vs. generated pitch visualisation.
- Sketch refinement pipeline (smoothing + normalisation).

---

## Repository Structure

```
.
├── app.py                        # Streamlit interactive demo
├── sketch_adapter.py             # Sketch refinement pipeline
├── fix_drawspeech.py             # Checkpoint key mismatch fix
├── preprocessing.py              # Feature extraction
├── compute_rmse.py               # RMSE evaluation
├── compute_rmse_10_utterances.py # 10‑utterance RMSE
├── build_tenx2.py                # 10×2 sketch builder
├── run_*.sbatch                  # SLURM experiment scripts
├── environment.yml               # Conda environment
├── RESULTS.md                    # Experiment results log
├── drawspeech/
│   ├── config/
│   │   └── drawspeech_ljspeech_22k.yaml
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
- [HiFi‑GAN](https://github.com/jik876/hifi-gan)

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

---

**Do you approve these changes?** If yes, I'll apply them to your README. If you'd like further adjustments, please let me know.
