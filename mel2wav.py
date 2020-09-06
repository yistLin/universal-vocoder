#!/usr/bin/env python3
"""Generate waveform from log mel spectrogram."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("npy_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    return vars(parser.parse_args())


def main(ckpt_path, npy_path, output_path):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(ckpt_path)
    model.to(device)

    mel = np.load(npy_path)
    mel = torch.FloatTensor(mel).to(device).transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        wav = model.generate(mel).squeeze().detach().cpu().numpy()

    npy_path_name = Path(npy_path).name
    wav_path = npy_path_name + ".wav" if output_path is None else output_path
    sf.write(wav_path, wav, model.sample_rate)


if __name__ == "__main__":
    main(**parse_args())
