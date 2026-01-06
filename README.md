<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Build guideline (on Linux x86 desktop PC):

```bash
mkdir MeloTTS_RK3588
cd MeloTTS_RK3588
python3 -m venv env
source env/bin/activate
pip install onnx onnxruntime
```

```bash
git clone git@github.com:kamejoko80/MeloTTS.git
git checkout henry_rk3588
cd ..
```

Install MeloTTS:

```bash
cd MeloTTS
pip install -e .
python -m unidic download
```

Run the bellow script to download nltk resource:

```bash
python - <<'PY'
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger_eng')
PY
```

Test torch inference:

```bash
cd MeloTTS/scripts
python test_torch.py
```

Export decoder:

```bash
python3 export_decoder.py --language EN --device cpu --T 128 --out models/decoder.onnx
```

Test decoder:

```bash
python3 test_decoder.py --decoder models/decoder.onnx --language EN --text "Did you ever hear a folk tale about a giant turtle?" --T 128 --out-pytorch pytorch_ref.wav --out-onnx onnx_decoder.wav
```

Export encoder:

```bash
python3 export_encoder.py --language EN --L 128 --device cpu --out models/encoder.onnx
```

Test encoder:

```bash
python3 test_encoder_decoder.py --encoder models/encoder.onnx --decoder models/decoder.onnx --language EN --text "Hello world, this should sound correct now." --speed 1.1 --out onnx_encoder_decoder.wav
```

Install RKNN-Toolkit2:

Must open a different linux terminal to install the RKNN-Toolkit2 on the Linux x86 desktop PC

```bash
cd MeloTTS_RK3588
mkdir RKNN-Toolkit2
cp MeloTTS/scripts/sh/Miniforge3-Linux-x86_64.sh ./RKNN-Toolkit2/Miniforge3-Linux-x86_64.sh
cd RKNN-Toolkit2
```

Run bash Miniforge3-Linux-x86_64.sh (availabe in the repo's scripts folder) and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.8 version:

```bash
conda create -n RKNN-Toolkit2 python=3.8
```

Activate RKNN-Toolkit2:

```bash
> conda activate RKNN-Toolkit2
```

To deactivate:

```bash
> conda deactivate
```

Install RKNN-Toolkit2 from github repo:

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2
pip install -r rknn-toolkit2/packages/x86_64/requirements_cp38-2.3.2.txt
pip install rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Covert ONNX to RKNN

```bash
cd MeloTTS/scripts
python3 convert.py --onnx models/encoder.onnx --out models/encoder.rknn --target rk3588 --opt 3 --fp16 --verbose
python3 convert.py --onnx models/decoder.onnx --out models/decoder.rknn --target rk3588 --opt 3 --fp16 --verbose
```

## Setup on RK3588:

```bash
mkdir MeloTTS
git clone https://github.com/kamejoko80/MeloTTS.git
cd MeloTTS
git checkout henry_rk3588
cd ..
cp MeloTTS/scripts/sh/Miniforge3-25.11.0-0-Linux-aarch64.sh ./
```

Run bash Miniforge3-25.11.0-0-Linux-aarch64.sh and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.10 version:

```bash
conda create -n RKNN-Toolkit2 python=3.10
```

Activate RKNN-Toolkit2:

```bash
> conda activate RKNN-Toolkit2
```

To deactivate:

```bash
> conda deactivate
```

Install RKNN-Toolkit2 & MeloTTS:

```bash
pip install rknn-toolkit-lite2
cd MeloTTS
pip install -e .
python -m unidic download
```

Run the bellow script to download nltk resource:

```bash
python - <<'PY'
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger_eng')
PY
```

Test torch inference:

```bash
cd MeloTTS/scripts
python test_torch.py
```

Test MeloTTS with RKNN accelerator:

Copy "encoder.rknn" & "decoder.rknn" from the Linux x86 PC into the MeloTTS/scripts/models folder, then run:

```bash
python3 test_melo_tts_rk3588.py --enc-rknn models/encoder.rknn --dec-rknn models/decoder.rknn --language EN --text "Hello world. RTF measurement on RK3588." --speed 1.1 --warmup 2 --speaker-id 1 --out rk3588_rtf.wav
```


## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
