#!/usr/bin/env python3
"""Reconstruct waveform from log mel spectrogram."""

from pathlib import Path

import torch
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile

from data import load_wav, log_mel_spectrogram


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("audio_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)

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
    audio_path,
    output_path,
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

    wav = load_wav(audio_path, sample_rate)
    mel = log_mel_spectrogram(
        wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min
    ).T

    mel = torch.FloatTensor(mel).to(device).transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        wav = model.generate(mel).squeeze().detach().cpu().numpy()

    npy_path_name = Path(audio_path).name
    wav_path = npy_path_name + ".rec.wav" if output_path is None else output_path
    sf.write(wav_path, wav, sample_rate)


if __name__ == "__main__":
    main(**parse_args())
