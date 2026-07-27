import os, json, argparse, numpy as np, librosa, pyworld as pw, tqdm

def extract_pitch_and_energy(wav_path, sr=22050, hop_length=256):
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=hop_length/sr*1000)
    f0 = pw.stonemask(wav, f0, t, sr)
    frame_length = 1024
    n_frames = len(f0)
    total = (n_frames - 1) * hop_length + frame_length
    wav_pad = np.pad(wav, (0, max(0, total - len(wav))))[:total]
    frames = librosa.util.frame(wav_pad, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    energy_db = 20 * np.log10(rms + 1e-8)
    return f0, energy_db[:n_frames]

def rmse_pair(a_path, b_path):
    ap, ae = extract_pitch_and_energy(a_path)
    bp, be = extract_pitch_and_energy(b_path)
    n = min(len(ap), len(bp))
    ap, bp, ae, be = ap[:n], bp[:n], ae[:n], be[:n]
    mask = (ap > 0) & (bp > 0)
    if mask.sum() < 3:
        return None
    return (np.sqrt(np.mean((ap[mask]-bp[mask])**2)),
            np.sqrt(np.mean((ae[mask]-be[mask])**2)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generated_dir', required=True)
    p.add_argument('--mapping', required=True,
                   help='JSON: {"generated_basename": "/path/to/reference.wav"}')
    a = p.parse_args()

    mapping = json.load(open(a.mapping))
    tp = te = 0.0
    count = skipped = 0
    for gen_name, ref_path in tqdm.tqdm(mapping.items()):
        gen_path = os.path.join(a.generated_dir, gen_name)
        if not (os.path.exists(gen_path) and os.path.exists(ref_path)):
            skipped += 1
            continue
        try:
            r = rmse_pair(gen_path, ref_path)
        except Exception as e:
            print(f"skip {gen_name}: {e}")
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        tp += r[0]; te += r[1]; count += 1

    if count:
        print(f"Processed {count} pairs (skipped {skipped})")
        print(f"Average Pitch RMSE: {tp/count:.2f} Hz")
        print(f"Average Energy RMSE: {te/count:.2f} dB")
    else:
        print(f"No valid pairs (skipped {skipped}).")

if __name__ == '__main__':
    main()