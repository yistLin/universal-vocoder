"""Vocoder dataset."""

import json
from random import randint
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import mulaw_encode


class VocoderDataset(Dataset):
    """Sample a segment of utterance for training vocoder."""

    def __init__(
        self, data_dir, metadata_path, frames_per_sample, frames_per_slice, bits
    ):

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.data_dir = Path(data_dir)
        self.sample_rate = metadata["sample_rate"]
        self.hop_len = metadata["hop_len"]
        self.n_mels = metadata["n_mels"]
        self.n_pad = (frames_per_sample - frames_per_slice) // 2
        self.frames_per_sample = frames_per_sample
        self.frames_per_slice = frames_per_slice
        self.bits = bits
        self.uttr_infos = [
            uttr_info
            for uttr_info in metadata["utterances"]
            if uttr_info["mel_len"] > frames_per_sample
        ]

    def __len__(self):
        return len(self.uttr_infos)

    def __getitem__(self, index):
        uttr_info = self.uttr_infos[index]
        features = np.load(self.data_dir / uttr_info["feature_path"])
        wav = features["wav"]
        mel = features["mel"]

        wav = np.pad(wav, (0, (len(mel) * self.hop_len - len(wav))), "constant")
        mel = np.pad(mel, ((self.n_pad,), (0,)), "constant")
        wav = np.pad(wav, (self.n_pad * self.hop_len,), "constant")
        wav = mulaw_encode(wav, 2 ** self.bits)

        pos = randint(0, len(mel) - self.frames_per_sample)
        mel_seg = mel[pos : pos + self.frames_per_sample, :]

        pos1 = pos + self.n_pad
        pos2 = pos1 + self.frames_per_slice
        wav_seg = wav[pos1 * self.hop_len : pos2 * self.hop_len + 1]

        return torch.FloatTensor(mel_seg), torch.LongTensor(wav_seg)
