#!/usr/bin/env python
"""Preprocess script"""

import os
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from itertools import chain
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from jsonargparse import ArgumentParser, ActionConfigFile
from librosa.util import find_files
from tqdm import tqdm

from data import load_wav, log_mel_spectrogram


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-w", "--n_workers", type=int, default=cpu_count())

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop_len", type=int, default=200)
    parser.add_argument("--win_len", type=int, default=800)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--f_min", type=int, default=50)
    parser.add_argument("--audio_config", action=ActionConfigFile)

    args = vars(parser.parse_args())

    return args


def load_process_save(
    audio_path, save_dir, sample_rate, preemph, hop_len, win_len, n_fft, n_mels, f_min,
):
    """Load an audio file, process, and save npz object."""

    wav = load_wav(audio_path, sample_rate)
    mel = log_mel_spectrogram(
        wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min
    )

    fd, temp_file_path = mkstemp(suffix=".npz", prefix="utterance-", dir=save_dir)
    np.savez_compressed(temp_file_path, wav=wav, mel=mel)
    os.close(fd)

    return {
        "feature_path": Path(temp_file_path).name,
        "audio_path": audio_path,
        "wav_len": len(wav),
        "mel_len": len(mel),
    }


def main(
    data_dirs,
    out_dir,
    n_workers,
    sample_rate,
    preemph,
    hop_len,
    win_len,
    n_fft,
    n_mels,
    f_min,
    **kwargs,
):
    """Preprocess audio files into features for training."""

    audio_paths = chain.from_iterable([find_files(data_dir) for data_dir in data_dirs])

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=n_workers)

    futures = []
    for audio_path in audio_paths:
        futures.append(
            executor.submit(
                load_process_save,
                audio_path,
                save_dir,
                sample_rate,
                preemph,
                hop_len,
                win_len,
                n_fft,
                n_mels,
                f_min,
            )
        )

    infos = {
        "sample_rate": sample_rate,
        "preemph": preemph,
        "hop_len": hop_len,
        "win_len": win_len,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "f_min": f_min,
        "utterances": [future.result() for future in tqdm(futures)],
    }

    with open(save_dir / "metadata.csv", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
