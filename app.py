import os
import glob
import re
import tempfile

import numpy as np
import torch
import yaml
import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import streamlit as st

from g2p_en import G2p
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from scipy.interpolate import PchipInterpolator

from drawspeech.utilities.model_util import instantiate_from_config
from drawspeech.utilities.data.dataset import AudioDataset
from sketch_adapter import SketchAdapter

adapter = SketchAdapter()


# -------------------- 1. Load model --------------------
@st.cache_resource
def load_model(config_yaml_path, checkpoint_path):
    with open(config_yaml_path) as f:
        config = yaml.safe_load(f)
    config["reload_from_ckpt"] = checkpoint_path
    seed_everything(0)

    latent_diffusion = instantiate_from_config(config["model"])
    latent_diffusion.set_log_dir(config["log_directory"], "streamlit", "demo")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    latent_diffusion.load_state_dict(state_dict, strict=True)

    for key in latent_diffusion.cond_stage_model_metadata.keys():
        model_idx = latent_diffusion.cond_stage_model_metadata[key]["model_idx"]
        model = latent_diffusion.cond_stage_models[model_idx]
        if hasattr(model, "infer"):
            model.infer = True

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()
    return latent_diffusion, config


# -------------------- 2. Synthesis --------------------
# NOTE: this is the original synthesis path from the first version of the app.
# Testing showed it does not exhibit the ghosting artifact seen with the
# frame-expansion / pitch-rescaling variant, so it is kept as-is.
def synthesize(model, config, text, pitch_sketch, energy_sketch, duration=None, phones_list=None):
    if phones_list is None:
        g2p = G2p()
        phones = g2p(text)
    else:
        phones = phones_list
    phoneme_str = "{" + " ".join(phones) + "}"
    n_phones = len(phones)

    def pad_sketch(sketch):
        if sketch is None:
            return None
        arr = np.array(sketch)
        if len(arr) < n_phones:
            arr = np.pad(arr, (0, n_phones - len(arr)), constant_values=0.5)
        else:
            arr = arr[:n_phones]
        return arr.tolist()

    pitch_sketch = pad_sketch(pitch_sketch)
    energy_sketch = pad_sketch(energy_sketch)
    if duration is None:
        duration = [10] * n_phones

    def save_temp(arr):
        if arr is None:
            return ""
        path = tempfile.mktemp(suffix=".npy")
        np.save(path, np.array(arr))
        return path

    pitch_path = save_temp(pitch_sketch)
    energy_path = save_temp(energy_sketch)

    sample = {
        "wav": "dummy.wav",
        "speaker": "LJSpeech",
        "transcription": text,
        "phonemes": phoneme_str,
        "pitch_sketch": pitch_path,
        "energy_sketch": energy_path,
        "pitch": "",
        "energy": "",
        "duration": duration,
        "pitch_length": n_phones if pitch_sketch is not None else 0,
        "energy_length": n_phones if energy_sketch is not None else 0,
    }
    dataset_json = {"data": [sample]}

    config["preprocessing"]["preprocessed_data"] = {
        "pitch": "data/dataset/metadata/ljspeech/phoneme_level/pitch",
        "energy": "data/dataset/metadata/ljspeech/phoneme_level/energy",
        "duration": "data/dataset/metadata/ljspeech/phoneme_level/duration",
        "stats_json": "data/dataset/metadata/ljspeech/phoneme_level/stats.json",
        "feature": "phoneme_level",
    }

    dataloader_add_ons = config["data"].get("dataloader_add_ons", [])
    dataset = AudioDataset(config=config, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json)
    loader = DataLoader(dataset, batch_size=1)

    eval_params = config["model"]["params"]["evaluation_params"]
    guidance_scale = eval_params["unconditional_guidance_scale"]
    ddim_steps = eval_params["ddim_sampling_steps"]
    n_gen = eval_params["n_candidates_per_samples"]

    with torch.no_grad():
        model.generate_sample(
            loader,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=n_gen,
        )

    demo_dir = os.path.join(config["log_directory"], "streamlit", "demo")
    infer_dirs = sorted(glob.glob(os.path.join(demo_dir, "infer_*")), key=os.path.getmtime)
    if infer_dirs:
        latest_dir = infer_dirs[-1]
        wav_files = sorted(glob.glob(os.path.join(latest_dir, "*.wav")), key=os.path.getmtime)
        if wav_files:
            return wav_files[-1], pitch_sketch, energy_sketch
    return None, pitch_sketch, energy_sketch


# -------------------- 3. Pitch extraction from generated WAV --------------------
def extract_pitch_from_wav(wav_path, n_phones):
    wav, sr = librosa.load(wav_path, sr=22050)
    wav = wav.astype(np.float64)
    hop_length = 256
    f0, t = pw.dio(wav, sr, frame_period=hop_length / sr * 1000)
    f0 = pw.stonemask(wav, f0, t, sr)
    nonzero = np.where(f0 > 0)[0]
    if len(nonzero) < 2:
        return None
    f0_interp = np.interp(np.arange(len(f0)), nonzero, f0[nonzero])
    x_old = np.linspace(0, 1, len(f0_interp))
    x_new = np.linspace(0, 1, n_phones)
    f0_phoneme = np.interp(x_new, x_old, f0_interp)
    f0_min, f0_max = f0_phoneme.min(), f0_phoneme.max()
    if f0_max - f0_min < 1e-6:
        return None
    return (f0_phoneme - f0_min) / (f0_max - f0_min)


# -------------------- 4. Word / phoneme parsing --------------------
def parse_words_and_phonemes(text, g2p):
    """Tokenize text into words and run G2P per-word, so phoneme grouping
    always matches word boundaries exactly (no silence-token guessing)."""
    words = re.findall(r"[A-Za-z']+", text)

    word_phone_lists = []
    phones = []
    for word in words:
        ph_list = g2p(word)
        # Drop punctuation-only tokens returned by g2p_en
        ph_list = [ph for ph in ph_list if any(ch.isalpha() for ch in ph)]
        word_phone_lists.append(ph_list)
        phones.extend(ph_list)

    word_phones = [len(ph_list) for ph_list in word_phone_lists]
    return words, phones, word_phone_lists, word_phones


def pchip_pitch_from_word_values(word_values, word_phones, n_phones):
    """Interpolate one pitch value per word into a per-phoneme pitch curve
    using PCHIP, anchoring each word's value at the center of its phonemes."""
    x_anchor, y_anchor = [], []
    current = 0
    for value, count in zip(word_values, word_phones):
        center = current + (count - 1) / 2
        x_anchor.append(center)
        y_anchor.append(value)
        current += count

    if x_anchor[0] > 0:
        x_anchor.insert(0, 0)
        y_anchor.insert(0, word_values[0])
    if x_anchor[-1] < n_phones - 1:
        x_anchor.append(n_phones - 1)
        y_anchor.append(word_values[-1])

    interp = PchipInterpolator(x_anchor, y_anchor)
    return interp(np.arange(n_phones))


# -------------------- 5. Plotting helpers --------------------
def plot_pitch_with_word_labels(pitch_sketch, phones, words, word_phones):
    fig, ax = plt.subplots(figsize=(max(6, len(phones) * 0.3), 3))
    ax.plot(pitch_sketch, marker="o")
    ax.set_xticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=8)

    cum = 0
    for word, count in zip(words, word_phones):
        ax.axvline(x=cum - 0.5, color="gray", linestyle="--", linewidth=0.5)
        ax.text(cum + count / 2 - 0.5, 1.05, word, ha="center", fontsize=10,
                 transform=ax.get_xaxis_transform())
        cum += count

    ax.set_ylim(0, 1)
    ax.set_ylabel("Pitch")
    return fig


# -------------------- 6. UI --------------------
def main():
    st.set_page_config(page_title="DrawSpeech Demo", layout="wide")
    st.title("DrawSpeech – Expressive Speech Synthesis with Pitch Sliders")
    st.markdown(
        "Control the intonation of synthesised speech using **word-level** or **phoneme-level** pitch sliders."
    )

    with st.spinner("Loading DrawSpeech model..."):
        model, config = load_model(
            "drawspeech/config/drawspeech_ljspeech_22k.yaml",
            "data/checkpoints/drawspeech_fixed.ckpt",
        )
    st.success("Model ready!")

    text = st.text_input("Enter text:", "I didn't say you stole the money.")
    g2p = G2p()
    words, phones, word_phone_lists, word_phones = parse_words_and_phonemes(text, g2p)
    n_phones = len(phones)

    st.write(f"**Words:** {' | '.join(words)}")
    st.write("**Phonemes per word:**")
    for word, ph_list in zip(words, word_phone_lists):
        st.write(f"**{word}** ({len(ph_list)}): {' '.join(ph_list)}")

    input_mode = st.radio(
        "Control granularity:",
        ["Word Pitch Sliders", "Phoneme Pitch Sliders"],
        horizontal=True,
    )

    pitch_sketch = None

    if input_mode == "Word Pitch Sliders":
        st.subheader("Word Pitch Sliders")
        st.caption("Set one pitch value per word. The value is repeated for all its phonemes.")

        word_values = []
        cols = st.columns(len(words))
        for i, (word, col) in enumerate(zip(words, cols)):
            with col:
                val = st.slider(word, 0.0, 1.0, 0.5, key=f"ws_{i}_{word}")
                word_values.append(val)

        if word_values:
            pitch_sketch = pchip_pitch_from_word_values(word_values, word_phones, n_phones)
            pitch_sketch = np.clip(pitch_sketch, 0.0, 1.0)
            pitch_sketch = adapter(pitch_sketch, word_values, word_phones)

            fig = plot_pitch_with_word_labels(pitch_sketch, phones, words, word_phones)
            st.pyplot(fig)

    elif input_mode == "Phoneme Pitch Sliders":
        st.subheader("Phoneme Pitch Sliders")
        st.caption("Set a pitch value for each phoneme. Words are grouped under their headings.")

        pitch_sketch = []
        word_start_idx = 0
        for word, ph_list in zip(words, word_phone_lists):
            count = len(ph_list)
            st.markdown(f"**{word}**")
            cols = st.columns(count)
            for j, col in enumerate(cols):
                idx = word_start_idx + j
                with col:
                    val = st.slider(phones[idx], 0.0, 1.0, 0.5, key=f"ps_{idx}")
                    pitch_sketch.append(val)
            word_start_idx += count

        if pitch_sketch:
            fig = plot_pitch_with_word_labels(pitch_sketch, phones, words, word_phones)
            st.pyplot(fig)

    energy_sketch = [0.5] * n_phones

    if st.button("Generate Speech"):
        if pitch_sketch is None:
            pitch_sketch = [0.5] * n_phones

        with st.spinner("Synthesizing..."):
            audio_path, used_pitch, used_energy = synthesize(
                model, config, text, pitch_sketch, energy_sketch, phones_list=phones
            )

        if not (audio_path and os.path.exists(audio_path)):
            st.error("Generation failed. Check terminal for details.")
            return

        st.audio(audio_path)
        st.success("Done!")
        with open(audio_path, "rb") as f:
            st.download_button("Download WAV", f, file_name="drawspeech_output.wav")

        
        # ---------------- Figure 2 ----------------

        sketch_arr = np.array(used_pitch)

        pred_arr = None
        if os.path.exists("predicted_contour.npy"):
            pred_arr = np.load("predicted_contour.npy").squeeze()

        synth_arr = None
        if os.path.exists("synthesized_pitch.npy"):
            synth_arr = np.load("synthesized_pitch.npy").squeeze()


        # ===============================
        # Figure 2
        # Sketch vs Predicted Contour
        # ===============================

        st.subheader("Figure 2: Sketch vs Predicted Contour")

        fig2, ax = plt.subplots(figsize=(max(8, n_phones * 0.4),4))

        ax.plot(
            sketch_arr,
            linewidth=3,
            label="Sketch"
        )

        if pred_arr is not None:

            L = min(len(pred_arr), len(sketch_arr))

            pred = pred_arr[:L]

            pred = (
                pred - pred.min()
            ) / (
                pred.max() - pred.min() + 1e-8
            )

            ax.plot(
                pred,
                "--",
                linewidth=2,
                label="Predicted contour"
            )

        cum = 0
        for word,count in zip(words,word_phones):

            if cum>0:
                ax.axvline(
                    x=cum-0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5
                )

            ax.text(
                cum+count/2-0.5,
                1.03,
                word,
                ha="center",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
            )

            cum+=count

        ax.set_xticks(np.arange(n_phones))
        ax.set_xticklabels(phones,rotation=90,fontsize=8)
        ax.set_ylabel("Normalized pitch")

        ax.legend()

        st.pyplot(fig2)


        # ===============================
        # Figure 3
        # Sketch vs Synthesized Pitch
        # ===============================

        st.subheader("Figure 3: Sketch vs Synthesized Pitch")

        fig3, ax = plt.subplots(figsize=(max(8, n_phones * 0.4),4))

        ax.plot(
            sketch_arr,
            linewidth=3,
            label="Sketch"
        )

        if synth_arr is not None:

            L = min(len(synth_arr),len(sketch_arr))

            synth = synth_arr[:L]

            synth = (
                synth - synth.min()
            ) / (
                synth.max()-synth.min()+1e-8
            )

            ax.scatter(
                np.arange(L),
                synth,
                s=18,
                label="Synthesized pitch"
            )

        cum = 0
        for word,count in zip(words,word_phones):

            if cum>0:
                ax.axvline(
                    x=cum-0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5
                )

            ax.text(
                cum+count/2-0.5,
                1.03,
                word,
                ha="center",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
            )

            cum+=count

        ax.set_xticks(np.arange(n_phones))
        ax.set_xticklabels(phones,rotation=90,fontsize=8)
        ax.set_ylabel("Normalized pitch")

        ax.legend()

        st.pyplot(fig3)


if __name__ == "__main__":
    main()