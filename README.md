# VQA Captioning

Installation and downloading of existing models, refer to [link](https://github.com/windweller/VQACaptioning/old_README.md).

Need to install KenLM for as the language modeling prior:

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

The `RSAModel` is added as a wrapper around original code.

A list of files/models that we modify:

- `models/CaptionModel` 
  - We added RSA-based sampling methods in there
  