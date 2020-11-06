# Universal Vocoder

This is a restructured and rewritten version of [bshall/UniversalVocoding](https://github.com/bshall/UniversalVocoding).
The main difference here is that the model is turned into a [TorchScript](https://pytorch.org/docs/stable/jit.html) module during training and can be loaded for inferencing anywhere without Python dependencies.

## Generate waveforms using pretrained models

Since the pretrained models were turned to TorchScript, you can load a trained model anywhere.
Also you can generate multiple waveforms parallelly, e.g.

```python
import torch

vocoder = torch.jit.load("vocoder.pt")

mels = [
    torch.randn(100, 80),
    torch.randn(200, 80),
    torch.randn(300, 80),
] # (length, mel_dim)

with torch.no_grad():
    wavs = vocoder.generate(mels)
```

Emperically, if you're using the default architecture, you can generate 30 samples at the same time on an GTX 1080 Ti.

## Train from scratch

Multiple directories containing audio files can be processed at the same time, e.g.

```bash
python preprocess.py \
    VCTK-Corpus \
    LibriTTS/train-clean-100 \
    preprocessed # the output directory of preprocessed data
```

And train the model with the preprocessed data, e.g.

```bash
python train.py preprocessed
```

With the default settings, it would take around 12 hr to train to 100K steps on an RTX 2080 Ti.

## References

- [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292)
