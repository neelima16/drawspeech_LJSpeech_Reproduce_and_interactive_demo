#!/usr/bin/env python3
"""
fix_drawspeech.py — patch upstream DrawSpeech for modern libs.
Run once from repo root after cloning. Idempotent.
    python fix_drawspeech.py
"""
import sys, os

PATCHES = [
    # 1. stft.py — librosa 0.10+ API
    dict(
        file="drawspeech/utilities/audio/stft.py",
        why="librosa>=0.10 made pad_center's size keyword-only",
        old="fft_window = pad_center(fft_window, filter_length)",
        new="fft_window = pad_center(fft_window, size=filter_length)",
    ),
    # 2. preprocessor.py — unbound variables
    dict(
        file="drawspeech/utilities/preprocessor/preprocessor.py",
        why="pitch/energy/n unbound when TextGrid missing",
        old="""            basename = wav_name.split(".")[0]""",
        new="""            pitch, energy, n = [], [], 0
            basename = wav_name.split(".")[0]""",
    ),
    # 3. ddpm.py — CLAP class undefined
    dict(
        file="drawspeech/modules/latent_diffusion/ddpm.py",
        why="CLAPAudioEmbeddingClassifierFreev2 undefined in this repo",
        old="""                if not self.training:
                    if isinstance(self.cond_stage_models[self.cond_stage_model_metadata[cond_model_key]["model_idx"]], CLAPAudioEmbeddingClassifierFreev2):
                        print("Warning: CLAP model normally should use text for evaluation")""",
        new="""                # PATCH: CLAP class absent; check removed""",
    ),
    # 4. ddpm.py — tkinter import on headless
    dict(
        file="drawspeech/modules/latent_diffusion/ddpm.py",
        why="tkinter import fails on headless HPC nodes",
        old="from tkinter import E\n",
        new="",
    ),
    # 5. infer.py — checkpoint format
    dict(
        file="drawspeech/infer.py",
        why="released checkpoints are bare state dicts",
        old='latent_diffusion.load_state_dict(checkpoint["state_dict"])',
        new='latent_diffusion.load_state_dict(checkpoint.get("state_dict", checkpoint))',
    ),
    # 6. conditional_models.py — _mel_mask crash
    dict(
        file="drawspeech/conditional_models.py",
        why="_mel_mask=None crashes downstream at inference",
        old="_mel_mask = None",
        new="_mel_mask = torch.zeros_like(mel_mask).bool()  # PATCH: was None",
    ),
    # 7. YAML config — VAE checkpoint path
    dict(
        file="drawspeech/config/drawspeech_ljspeech_22k.yaml",
        why="point at the downloaded VAE checkpoint",
        old='reload_from_ckpt: "log/latent_diffusion/vae_ljspeech_22k/checkpoints/checkpoint-79999.ckpt"',
        new='reload_from_ckpt: "data/checkpoints/vae.ckpt"',
    ),
]

def main():
    if not os.path.isdir("drawspeech"):
        sys.exit("Run this from the repo root (no ./drawspeech found).")

    applied = skipped = failed = 0
    for p in PATCHES:
        f, tag = p["file"], f"{p['file']}: {p['why']}"
        if not os.path.exists(f):
            print(f"[FAIL ] {tag}\n         file not found"); failed += 1; continue
        
        src = open(f, encoding="utf-8").read()
        
        # Already applied?
        if p["new"] and p["new"] in src:
            print(f"[SKIP ] {tag}"); skipped += 1; continue
        
        # Can we find the old text?
        if p["old"] not in src:
            # For _mel_mask, try to find it manually
            if "_mel_mask = None" in p["old"] and "_mel_mask = None" in src:
                # Replace all occurrences
                count = src.count("_mel_mask = None")
                src = src.replace("_mel_mask = None", "_mel_mask = torch.zeros_like(mel_mask).bool()  # PATCH: was None")
                open(f, "w", encoding="utf-8").write(src)
                print(f"[APPLY] {tag} ({count} occurrences)"); applied += 1; continue
            else:
                print(f"[FAIL ] {tag}\n         target text not found — inspect manually"); failed += 1; continue
        
        # Apply the patch
        open(f, "w", encoding="utf-8").write(src.replace(p["old"], p["new"], 1))
        print(f"[APPLY] {tag}"); applied += 1

    # Extra: fix checkpoint keys if drawspeech.ckpt exists
    ckpt_path = "data/checkpoints/drawspeech.ckpt"
    fixed_path = "data/checkpoints/drawspeech_fixed.ckpt"
    if os.path.exists(ckpt_path) and not os.path.exists(fixed_path):
        print("\n=== Fixing checkpoint key names ===")
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        new_state = {}
        renamed = 0
        for k, v in ckpt.items():
            new_k = k.replace("phoneme_encoder", "text_encoder")
            new_k = new_k.replace("detailed_curve_predictor", "sketch_to_contour_predictor")
            if new_k != k:
                renamed += 1
            new_state[new_k] = v
        torch.save(new_state, fixed_path)
        print(f"Renamed {renamed} keys. Fixed checkpoint saved to {fixed_path}")

    print(f"\napplied={applied} skipped={skipped} failed={failed}")
    if failed:
        print("Some patches failed. Upstream may have changed — patch those by hand.")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())