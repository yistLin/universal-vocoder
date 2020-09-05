#!/usr/bin/env python3
"""Train reconstruction model."""

from datetime import datetime
from pathlib import Path

import tqdm
import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from jsonargparse import ArgumentParser, ActionConfigFile

from data import VocoderDataset
from models import Vocoder


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("metadata_path", type=str)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--comment", type=str)

    parser.add_argument("--frames_per_sample", type=int, default=40)
    parser.add_argument("--frames_per_slice", type=int, default=8)
    parser.add_argument("--bits", type=int, default=9)
    parser.add_argument("--conditioning_channels", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--rnn_channels", type=int, default=896)
    parser.add_argument("--fc_channels", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--valid_every", type=int, default=1000)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--training_config", action=ActionConfigFile)

    return parser.parse_args()


def main(
    data_dir,
    metadata_path,
    n_workers,
    comment,
    frames_per_sample,
    frames_per_slice,
    bits,
    conditioning_channels,
    embedding_dim,
    rnn_channels,
    fc_channels,
    batch_size,
    n_steps,
    valid_every,
    valid_ratio,
    save_every,
    save_dir,
    **kwargs,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VocoderDataset(
        data_dir, metadata_path, frames_per_sample, frames_per_slice, bits
    )
    lengths = [
        trainlen := int((1 - valid_ratio) * len(dataset)),
        len(dataset) - trainlen,
    ]
    trainset, validset = random_split(dataset, lengths)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    model = Vocoder(
        sample_rate=dataset.sample_rate,
        mel_channels=dataset.n_mels,
        conditioning_channels=conditioning_channels,
        embedding_dim=embedding_dim,
        rnn_channels=rnn_channels,
        fc_channels=fc_channels,
        bits=bits,
        hop_length=dataset.hop_len,
    )
    model.to(device)

    optimizer = Adam(model.parameters())

    if comment is not None:
        log_dir = "logs/"
        log_dir += datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_dir += "_" + comment
        writer = SummaryWriter(log_dir)

    train_iterator = iter(train_loader)
    losses = []
    pbar = tqdm.tqdm(total=valid_every * train_loader.batch_size)

    for step in range(n_steps):
        try:
            mels, wavs = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            mels, wavs = next(train_iterator)

        mels = mels.to(device)
        wavs = wavs.to(device)

        outs = model(wavs[:, :-1], mels)

        loss = cross_entropy(outs.transpose(1, 2), wavs[:, 1:])

        losses.append(loss.item())
        pbar.set_description(f"step = {step+1}, loss = {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(train_loader.batch_size)

        if (step + 1) % valid_every == 0:
            pbar.close()

            train_loss = sum(losses) / len(losses)
            print(f"[train] loss = {train_loss:.4f}")
            losses = []

            pbar = tqdm.tqdm(total=len(valid_loader.dataset), leave=False)
            for mels, wavs in valid_loader:
                mels = mels.to(device)
                wavs = wavs.to(device)
                with torch.no_grad():
                    outs = model(wavs[:, :-1], mels)
                    loss = cross_entropy(outs.transpose(1, 2), wavs[:, 1:])
                losses.append(loss.item())
                pbar.update(valid_loader.batch_size)
            pbar.close()

            valid_loss = sum(losses) / len(losses)
            print(f"[valid] loss = {valid_loss:.4f}")
            losses = []

            if comment is not None:
                writer.add_scalar("Loss/train", train_loss, step + 1)
                writer.add_scalar("Loss/valid", valid_loss, step + 1)

            pbar = tqdm.tqdm(total=valid_every * train_loader.batch_size)

        if (step + 1) % save_every == 0:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir_path / f"vocoder-ckpt-{step+1}.pt"
            model.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main(**vars(parse_args()))
