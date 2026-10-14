"""Split music11.mp3 into stems with Demucs (htdemucs), writing WAVs to analysis/stems/.

Usage: .venv/Scripts/python.exe tools/separate.py <input audio> <out dir>
Audio is decoded with ffmpeg so torchaudio's I/O backends are never needed.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def load(path, sr):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = get_model("htdemucs")
    model.eval()
    wav = torch.from_numpy(load(src, model.samplerate))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.no_grad():
        stems = apply_model(model, ((wav - mean) / std)[None], shifts=1, split=True,
                            overlap=0.25, progress=True, num_workers=0)[0]
    stems = stems * std + mean
    for name, s in zip(model.sources, stems):
        sf.write(out / f"{src.stem}_{name}.wav", s.numpy().T, model.samplerate, subtype="PCM_16")
        print("wrote", out / f"{src.stem}_{name}.wav")


if __name__ == "__main__":
    main()
