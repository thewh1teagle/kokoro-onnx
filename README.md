# kokoro-onnx

TTS with onnx runtime based on [Kokoro-TTS](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

## Features

- Support `English`, `French`, `Japanse`, `Korean`, and `Chinese`
- 4X Faster than realtime (macOS M1)
- Support multiple voices including whispering

## Setup

```console
pip install -U kokoro-onnx
```

- You also need [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx), and [`voices.json`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json). 
- We recommend use [uv](https://docs.astral.sh/uv/getting-started/installation) for Python.
- Python 3.12 is recommended. (`uv init -p 3.12`)
- Please see examples.

## Examples

See [examples](examples)

## Voices

Available voices are `af`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

<video src="https://github.com/user-attachments/assets/a89b4c75-303d-47ac-96c8-7edb64b9150a" width=400></video>
