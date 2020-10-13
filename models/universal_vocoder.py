"""Universal vocoder"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class UniversalVocoder(nn.Module):
    """Universal vocoding"""

    def __init__(
        self,
        sample_rate,
        frames_per_sample,
        frames_per_slice,
        mel_dim,
        mel_rnn_dim,
        emb_dim,
        wav_rnn_dim,
        affine_dim,
        bits,
        hop_length,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.frames_per_slice = frames_per_slice
        self.pad = (frames_per_sample - frames_per_slice) // 2
        self.wav_rnn_dim = wav_rnn_dim
        self.quant_dim = 2 ** bits
        self.hop_len = hop_length

        self.mel_rnn = nn.GRU(
            mel_dim, mel_rnn_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.embedding = nn.Embedding(self.quant_dim, emb_dim)
        self.wav_rnn = nn.GRU(emb_dim + 2 * mel_rnn_dim, wav_rnn_dim, batch_first=True)
        self.affine = nn.Sequential(
            nn.Linear(wav_rnn_dim, affine_dim),
            nn.ReLU(),
            nn.Linear(affine_dim, self.quant_dim),
        )

    def forward(self, wavs, mels):
        """Generate waveform from mel spectrogram with teacher-forcing."""
        mel_embs, _ = self.mel_rnn(mels)
        mel_embs = mel_embs.transpose(1, 2)
        mel_embs = mel_embs[:, :, self.pad : self.pad + self.frames_per_slice]

        conditions = F.interpolate(mel_embs, scale_factor=float(self.hop_len))
        conditions = conditions.transpose(1, 2)

        wav_embs = self.embedding(wavs)
        wav_outs, _ = self.wav_rnn(torch.cat((wav_embs, conditions), dim=2))

        return self.affine(wav_outs)

    @torch.jit.export
    def generate(self, mels: List[Tensor]) -> List[Tensor]:
        """Generate waveform from mel spectrogram.

        Args:
            mels: list of tensor of shape (mel_len, mel_dim)

        Returns:
            wavs: list of tensor of shape (wav_len)
        """

        batch_size = len(mels)
        device = mels[0].device

        mel_lens = [len(mel) for mel in mels]
        wav_lens = [mel_len * self.hop_len for mel_len in mel_lens]
        max_mel_len = max(mel_lens)
        max_wav_len = max_mel_len * self.hop_len

        pad_mels = pad_sequence(mels, batch_first=True)
        pack_mels = pack_padded_sequence(
            pad_mels, torch.tensor(mel_lens), batch_first=True, enforce_sorted=False
        )
        pack_mel_embs, _ = self.mel_rnn(pack_mels)
        mel_embs, _ = pad_packed_sequence(
            pack_mel_embs, batch_first=True
        )  # (batch, max_mel_len, emb_dim)

        mel_embs = mel_embs.transpose(1, 2)
        conditions = F.interpolate(mel_embs, scale_factor=float(self.hop_len))
        conditions = conditions.transpose(1, 2)  # (batch, max_wav_len, emb_dim)

        hid = torch.zeros(1, batch_size, self.wav_rnn_dim, device=device)
        wav = torch.full(
            (batch_size,), self.quant_dim // 2, dtype=torch.long, device=device,
        )
        wavs = torch.empty(batch_size, max_wav_len, dtype=torch.float, device=device,)

        for i, condition in enumerate(torch.unbind(conditions, dim=1)):
            wav_emb = self.embedding(wav)
            wav_rnn_input = torch.cat((wav_emb, condition), dim=1).unsqueeze(1)
            _, hid = self.wav_rnn(wav_rnn_input, hid)
            logit = self.affine(hid.squeeze(0))
            posterior = F.softmax(logit, dim=1)
            wav = torch.multinomial(posterior, 1).squeeze(1)
            wavs[:, i] = 2 * wav / (self.quant_dim - 1.0) - 1.0

        mu = self.quant_dim - 1
        wavs = torch.true_divide(torch.sign(wavs), mu) * (
            (1 + mu) ** torch.abs(wavs) - 1
        )
        wavs = [
            wav[:length] for wav, length in zip(torch.unbind(wavs, dim=0), wav_lens)
        ]

        return wavs
