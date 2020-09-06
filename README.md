# Universal Vocoder

This is a restructured and rewritten version of [bshall/UniversalVocoding](https://github.com/bshall/UniversalVocoding).
The main difference here is that the model is turned into a [TorchScript](https://pytorch.org/docs/stable/jit.html) module during training and can be loaded for inferencing anywhere without Python dependencies.

### References

- [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292)
