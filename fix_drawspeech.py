#!/usr/bin/env python3
"""
fix_drawspeech.py — patch upstream DrawSpeech for modern libs.
Run once from repo root after cloning. Idempotent.
    python fix_drawspeech.py
"""
import sys, os

PATCHES = [
    dict(
        file="drawspeech/utilities/audio/stft.py",
        why="librosa>=0.10 made pad_center's size keyword-only",
        old="fft_window = pad_center(fft_window, filter_length)",
        new="fft_window = pad_center(fft_window, size=filter_length)",
    ),
    dict(
        file="drawspeech/utilities/preprocessor/preprocessor.py",
        why="pitch/energy/n unbound when TextGrid missing",
        old="""        for wav_name in tqdm(os.listdir(self.in_dir)):
            if ".wav" not in wav_name:
                continue

            basename = wav_name.split(".")[0]""",
        new="""        for wav_name in tqdm(os.listdir(self.in_dir)):
            if ".wav" not in wav_name:
                continue

            # PATCH: init so a missing TextGrid can't leave these unbound
            pitch, energy, n = [], [], 0

            basename = wav_name.split(".")[0]""",
    ),
    dict(
        file="drawspeech/modules/latent_diffusion/ddpm.py",
        why="CLAPAudioEmbeddingClassifierFreev2 undefined in this repo",
        old="""                if not self.training:
                    if isinstance(self.cond_stage_models[self.cond_stage_model_metadata[cond_model_key]["model_idx"]], CLAPAudioEmbeddingClassifierFreev2):
                        print("Warning: CLAP model normally should use text for evaluation")""",
        new="""                # PATCH: CLAP class absent in DrawSpeech; check removed
                # if not self.training:
                #     if isinstance(self.cond_stage_models[self.cond_stage_model_metadata[cond_model_key]["model_idx"]], CLAPAudioEmbeddingClassifierFreev2):
                #         print("Warning: CLAP model normally should use text for evaluation")""",
    ),
    dict(
        file="drawspeech/modules/latent_diffusion/ddpm.py",
        why="tkinter import fails on headless HPC nodes",
        old="from tkinter import E\n",
        new="",
    ),
    dict(
        file="drawspeech/infer.py",
        why="released checkpoints are bare state dicts",
        old='latent_diffusion.load_state_dict(checkpoint["state_dict"])',
        new='latent_diffusion.load_state_dict(checkpoint.get("state_dict", checkpoint))',
    ),
    dict(
        file="drawspeech/conditional_models.py",
        why="_mel_mask=None crashes downstream at inference",
        old="""        elif self.infer:
            _mel_mask = None
            pitch = pitch if isinstance(pitch, torch.Tensor) else None""",
        new="""        elif self.infer:
            _mel_mask = torch.zeros_like(mel_mask).bool()  # PATCH: was None
            pitch = pitch if isinstance(pitch, torch.Tensor) else None""",
    ),
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
        if p["new"] and p["new"] in src:
            print(f"[SKIP ] {tag}"); skipped += 1; continue
        if p["old"] not in src:
            print(f"[FAIL ] {tag}\n         target text not found — inspect manually"); failed += 1; continue
        open(f, "w", encoding="utf-8").write(src.replace(p["old"], p["new"], 1))
        print(f"[APPLY] {tag}"); applied += 1

    print(f"\napplied={applied} skipped={skipped} failed={failed}")
    if failed:
        print("Some patches failed. Upstream may have changed — patch those by hand.")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
