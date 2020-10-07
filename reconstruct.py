#!/usr/bin/env python3
"""Reconstruct waveform from log mel spectrogram."""

from warnings import filterwarnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import torch
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile

from data import load_wav, log_mel_spectrogram


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("audio_paths", type=str, nargs="+")
    parser.add_argument("-o", "--output_dir", type=str, default=".")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop_len", type=int, default=200)
    parser.add_argument("--win_len", type=int, default=800)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--f_min", type=int, default=50)
    parser.add_argument("--audio_config", action=ActionConfigFile)
    return vars(parser.parse_args())


def main(
    ckpt_path,
    audio_paths,
    output_dir,
    sample_rate,
    preemph,
    hop_len,
    win_len,
    n_fft,
    n_mels,
    f_min,
    **kwargs,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(ckpt_path)
    model.to(device)

    path2wav = partial(load_wav, sample_rate=sample_rate)
    wav2mel = partial(
        log_mel_spectrogram,
        preemph=preemph,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        f_min=f_min,
    )

    with Pool(cpu_count()) as pool:
        wavs = pool.map(path2wav, audio_paths)
        mels = pool.map(wav2mel, wavs)

    print("mels length:", [len(mel) for mel in mels])

    mel_tensors = [torch.FloatTensor(mel).to(device) for mel in mels]

    with torch.no_grad():
        wavs = model.generate(mel_tensors)
        wavs = [wav.detach().cpu().numpy() for wav in wavs]

    for wav, audio_path in zip(wavs, audio_paths):
        wav_path_name = Path(audio_path).name
        wav_path = Path(output_dir, wav_path_name).with_suffix(".rec.wav")
        sf.write(wav_path, wav, sample_rate)


if __name__ == "__main__":
    filterwarnings("ignore")
    main(**parse_args())
