# DrawSpeech Reproduction — Results Log

## Environment
- torch 2.0.1+cu117, pytorch_lightning 1.9.5, python 3.10
- node: rtx3080 (tg080)

## Run 0 — Smoke test (pipeline validation)
- date: 2026-07-18
- checkpoint: data/checkpoints/drawspeech_fixed.ckpt (renamed keys, zero training)
- config: drawspeech/config/drawspeech_ljspeech_22k.yaml
- JSON: tests/inference.json (contour-conditioned, LJ001-0001 pitch/energy)
- seed: 0 (infer.py default)
- result: test_money.wav produced, full pipeline verified ✓

## Run 1 — Pitch cross-utterance (clean rebuild)
- date: 2026-07-18
- checkpoint: drawspeech_fixed.ckpt (zero training)
- config: drawspeech_ljspeech_22k.yaml | seed: 42 (pairing), 0 (inference)
- 300 test files, 0 skipped
- Pitch RMSE: 34.18 Hz | Energy RMSE: 4.54 dB
- note: old repo reported 57.67 under a non-reproducible setup (mixed
  dataset_plugin versions, SKETCH_MODE switch). 34.18 is the value from
  the clean documented pipeline. Interpretation pending self-recon control.

## Note: DRAWSPEECH_NO_DURATION flag
Added to dataset_plugin.py (get_preprocessed_meta). When env var =1,
forces duration="" so the model predicts its own timing. Default (unset)
= normal behavior. Used to test duration's effect on cross-utterance RMSE.
