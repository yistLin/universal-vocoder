# Universal Vocoder

This is a restructured and rewritten version of [bshall/UniversalVocoding](https://github.com/bshall/UniversalVocoding).
The main difference here is that the model is turned into a [TorchScript](https://pytorch.org/docs/stable/jit.html) module during training and can be loaded for inferencing anywhere without Python dependencies.

### Preprocess training data

Multiple directories containing audio files can be processed at the same time.

```bash
python preprocess.py VCTK-Corpus LibriTTS/train-clean-100 preprocessed
```

### Train from scratch

```bash
python train.py preprocessed preprocessed/metadata.json
```

### Generate waveforms

You can load a trained model anywhere and generate multiple waveforms parallelly.

```python
import torch

vocoder = torch.jit.load("vocoder.pt")
mels = [
    torch.randn(100, 80),
    torch.randn(200, 80),
    torch.randn(300, 80),
]
wavs = vocoder.generate(mels)
```

Emperically, if you're using the default architecture, you can generate 100 samples at the same time on an Nvidia GTX 1080 Ti.

### References

- [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292)
