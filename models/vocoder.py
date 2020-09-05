import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Vocoder(nn.Module):
    """Universal vocoding"""

    def __init__(
        self,
        sample_rate,
        mel_channels,
        conditioning_channels,
        embedding_dim,
        rnn_channels,
        fc_channels,
        bits,
        hop_length,
    ):
        super().__init__()

        self.init_params = {
            "sample_rate": sample_rate,
            "mel_channels": mel_channels,
            "conditioning_channels": conditioning_channels,
            "embedding_dim": embedding_dim,
            "rnn_channels": rnn_channels,
            "fc_channels": fc_channels,
            "bits": bits,
            "hop_length": hop_length,
        }

        self.rnn_channels = rnn_channels
        self.quantization_channels = 2 ** bits
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.rnn1 = nn.GRU(
            mel_channels,
            conditioning_channels,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn2 = nn.GRU(
            embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True
        )
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, mels):
        sample_frames = mels.size(1)
        audio_slice_frames = x.size(1) // self.hop_length
        pad = (sample_frames - audio_slice_frames) // 2

        mels, _ = self.rnn1(mels)
        mels = mels[:, pad : pad + audio_slice_frames, :]

        mels = F.interpolate(mels.transpose(1, 2), scale_factor=float(self.hop_length))
        mels = mels.transpose(1, 2)

        x = self.embedding(x)

        x, _ = self.rnn2(torch.cat((x, mels), dim=2))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = cls(**ckpt["init_params"])
        model.load_state_dict(ckpt["model"])
        return model

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {"init_params": self.init_params, "model": self.state_dict(),},
            checkpoint_path,
        )

    def generate(self, mel):
        """Generate waveform from mel spectrogram using vocoder."""
        output = []
        cell = get_gru_cell(self.rnn2)

        with torch.no_grad():
            mel, _ = self.rnn1(mel)

            mel = F.interpolate(
                mel.transpose(1, 2), scale_factor=float(self.hop_length)
            )
            mel = mel.transpose(1, 2)

            batch_size, _, _ = mel.size()

            h = torch.zeros(batch_size, self.rnn_channels, device=mel.device)
            x = (
                torch.zeros(batch_size, device=mel.device)
                .fill_(self.quantization_channels // 2)
                .long()
            )

            for m in tqdm(torch.unbind(mel, dim=1), leave=False):
                x = self.embedding(x)
                h = cell(torch.cat((x, m), dim=1), h)

                x = F.relu(self.fc1(h))
                logits = self.fc2(x)

                posterior = F.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(posterior)

                x = dist.sample()
                output.append(
                    2 * x.float().item() / (self.quantization_channels - 1.0) - 1.0
                )

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)

        return output


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


def mulaw_decode(x_mu: np.ndarray, n_channels: int) -> np.ndarray:
    """Decode mu-law encoded signal."""
    mu = n_channels - 1
    x = np.sign(x_mu) / mu * ((1 + mu) ** np.abs(x_mu) - 1)
    return x
