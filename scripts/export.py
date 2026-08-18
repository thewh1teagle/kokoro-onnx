# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kokoro==0.8.4",
#     "numpy",
#     "onnx>=1.17.0",
#     "onnxruntime>=1.20.1",
#     "onnxscript>=0.5.0",
# ]
#
# ///

"""
Export a Kokoro checkpoint to ONNX, with waveform and duration outputs.

Based on https://github.com/hexgrad/kokoro/blob/3f9dd88/examples/export.py

English (v1.0):
    mkdir -p checkpoints
    wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json -O checkpoints/config.json
    wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth -O checkpoints/kokoro-v1_0.pth
    uv run scripts/export.py -c checkpoints/config.json -p checkpoints/kokoro-v1_0.pth -o kokoro-v1.0.onnx

Chinese (v1.1-zh):
    wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json -O checkpoints/config-zh.json
    wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth -O checkpoints/kokoro-v1_1-zh.pth
    uv run scripts/export.py -c checkpoints/config-zh.json -p checkpoints/kokoro-v1_1-zh.pth -o kokoro-v1.1-zh.onnx

The duration output reports how many frames each token takes, 600 samples each,
which is what kokoro_onnx uses for timestamps and for sliding window synthesis.
config.json is embedded in the graph metadata, so the vocabulary travels with
the model and no longer has to match the copy shipped in the package.
Speed must stay a float input: exporting it from an int tensor makes the graph
truncate fractional speeds, so speed=0.9 becomes 0 and the model fails.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from kokoro import KModel
from kokoro.model import KModelForONNX

OPSET = 17
SAMPLE_RATE = 24000
FRAME = 600
CONFIG_KEY = "kokoro_config"


SAMPLE_PHONEMES = "həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst."


def sample_style(voice: str, length: int) -> torch.Tensor:
    """The real style vector for `length` phonemes, so parity is measured in
    distribution; a random one sends the model far off it."""
    # Chinese voices live in the v1.1-zh repository, everything else in v1.0
    repo = (
        "hexgrad/Kokoro-82M-v1.1-zh" if voice.startswith("z") else "hexgrad/Kokoro-82M"
    )
    try:
        from huggingface_hub import hf_hub_download

        pack = torch.load(
            hf_hub_download(repo, f"voices/{voice}.pt"), weights_only=True
        )
        return pack[length - 1]
    except Exception as error:  # offline, or the voice is not in that repo
        print(f"  could not load voice {voice} ({error}), using a random style")
        return torch.randn(1, 256) * 0.2


def sample_inputs(model: KModelForONNX, voice: str) -> tuple[torch.Tensor, ...]:
    """Inputs shaped like a real call, used to trace and to verify the graph."""
    vocab = model.kmodel.vocab
    ids = [vocab[p] for p in SAMPLE_PHONEMES if p in vocab]
    return (
        torch.LongTensor([[0, *ids, 0]]),
        sample_style(voice, len(ids)),
        torch.FloatTensor([1.0]),
    )


def embed_config(path: Path, config: Path) -> None:
    """Carry config.json inside the graph so the vocabulary travels with it."""
    graph = onnx.load(str(path))
    entry = graph.metadata_props.add()
    entry.key = CONFIG_KEY
    entry.value = json.dumps(json.loads(config.read_text(encoding="utf-8")))
    onnx.save(graph, str(path))
    print(f"embedded {config} as metadata key {CONFIG_KEY!r}")


def export(model: KModelForONNX, path: Path, voice: str) -> None:
    torch.onnx.export(
        model,
        args=sample_inputs(model, voice),
        f=str(path),
        export_params=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "duration"],
        opset_version=OPSET,
        dynamic_axes={
            "input_ids": {1: "input_ids_len"},
            "waveform": {0: "num_samples"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    print(f"exported {path} ({path.stat().st_size / 1e6:.0f} MB)")


def verify(model: KModelForONNX, path: Path, voice: str) -> None:
    """Check the graph against the torch model it was traced from."""
    inputs = sample_inputs(model, voice)
    with torch.no_grad():
        expected, expected_duration = model(*inputs)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    waveform, duration = session.run(
        None, dict(zip(names, (tensor.numpy() for tensor in inputs)))
    )

    reference = expected.numpy()
    frames = int(np.sum(duration))
    error = np.abs(waveform - reference).max() / np.abs(reference).max()
    print(f"  waveform  {len(waveform)} samples, {len(waveform) / SAMPLE_RATE:.2f}s")
    print(
        f"  duration  {frames} frames, {len(waveform) / frames:.0f} samples per frame"
    )
    print(
        f"  against torch  peak error {error:.1%}, "
        f"correlation {np.corrcoef(waveform, reference)[0, 1]:.4f}"
    )
    assert np.array_equal(duration, expected_duration.numpy()), "durations disagree"
    assert len(waveform) == frames * FRAME, f"expected {FRAME} samples per frame"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Kokoro model to ONNX")
    parser.add_argument("-c", "--config", default="checkpoints/config.json")
    parser.add_argument("-p", "--checkpoint", default="checkpoints/kokoro-v1_0.pth")
    parser.add_argument("-o", "--output", default="kokoro.onnx")
    parser.add_argument(
        "-v", "--voice", default="af_heart", help="voice used to verify"
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = KModelForONNX(
        KModel(config=args.config, model=args.checkpoint, disable_complex=True)
    ).eval()

    export(model, output, args.voice)
    embed_config(output, Path(args.config))
    if not args.skip_verify:
        verify(model, output, args.voice)


if __name__ == "__main__":
    main()
