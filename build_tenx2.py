import os, json, random, numpy as np
from scipy.signal import savgol_filter

random.seed(42)
root = 'data/dataset/metadata/ljspeech'
test = json.load(open(f'{root}/test.json'))['data'][:10]
test_names = [os.path.splitext(os.path.basename(i['wav']))[0] for i in test]

pool = []
for split in ['train.json', 'val.json']:
    for i in json.load(open(f'{root}/{split}'))['data']:
        b = os.path.splitext(os.path.basename(i['wav']))[0]
        if b not in test_names:
            pool.append(b)

os.makedirs('tests/tenx2/sketches', exist_ok=True)

def make_sketch(src_name, n_target, out_path):
    p = f'{root}/phoneme_level/pitch/LJSpeech-pitch-{src_name}.npy'
    if not os.path.exists(p):
        return False
    src = np.load(p)
    w = min(5, len(src))
    if w % 2 == 0: w += 1
    sk = savgol_filter(src, window_length=w, polyorder=2) if w >= 3 else src
    sk = np.interp(np.linspace(0,1,n_target), np.linspace(0,1,len(sk)), sk)
    np.save(out_path, sk)
    return True

for variant in ['A', 'B']:
    data, mapping = [], {}
    for item in test:
        b = os.path.splitext(os.path.basename(item['wav']))[0]
        n_ph = len(item.get('phonemes','').strip('{}').split())
        if n_ph == 0: continue
        src = random.choice([x for x in pool if x != b])
        sk_path = f'tests/tenx2/sketches/{b}_{variant}.npy'
        if not make_sketch(src, n_ph, sk_path): continue
        data.append({
            'wav': item['wav'],
            'transcription': item.get('transcription',''),
            'phonemes': item.get('phonemes',''),
            'pitch_sketch': sk_path,
            'energy_sketch': f'{root}/phoneme_level/energy/LJSpeech-energy-{b}.npy',
            'pitch': '', 'energy': '',
            'duration': f'{root}/phoneme_level/duration/LJSpeech-duration-{b}.npy',
        })
        mapping[f'{b}.wav'] = f'data/dataset/LJSpeech-1.1/wavs/{src}.wav'
    json.dump({'data': data}, open(f'tests/tenx2/inference_{variant}.json','w'), indent=2)
    json.dump(mapping, open(f'tests/tenx2/map_{variant}.json','w'), indent=2)
    print(f"variant {variant}: {len(data)} samples")
