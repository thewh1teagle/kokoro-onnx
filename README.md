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

- You also need to place the files [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx), and [`voices.json`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json) in the project folder.
- Please see examples.

<details>

<summary>Instructions</summary>

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation) for isolated Python (Recommend).

Basically open the terminal (PowerShell / Bash) and run the command listed in their website.

_Note: you don't have to use `uv`. but it just make things much simpler. You can use regular Python as well._

1. Create new project folder (you name it)
2. Prepare the environment and run in the project folder

```console
uv init -p 3.12
uv add kokoro-onnx soundfile
```

4. Paste the contents of [`examples/save.py`](https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py) in `hello.py`
5. Downloads the files [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx), and [`voices.json`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices) and place them in the same directory.
6. Run

```console
uv run hello.py
```

You can edit the text in `hello.py`

That's it! `audio.wav` should be created.

</details>

## Examples

See [examples](examples)

## Voices

Available voices are `af`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

<video src="https://github.com/user-attachments/assets/a89b4c75-303d-47ac-96c8-7edb64b9150a" width=400></video>

## Contribute

See [CONTRIBUTE.md](CONTRIBUTE.md)
