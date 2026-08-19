# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kokoro==0.8.4",
#     "numpy",
#     "coloredlogs",
#     "onnx>=1.17.0",
#     "onnxruntime>=1.20.1",
#     "psutil",
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
    uv run scripts/export.py -c checkpoints/config.json -p checkpoints/kokoro-v1_0.pth -o kokoro-v1.0.onnx --fp16 --int8

Chinese (v1.1-zh):
    wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json -O checkpoints/config-zh.json
    wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth -O checkpoints/kokoro-v1_1-zh.pth
    uv run scripts/export.py -c checkpoints/config-zh.json -p checkpoints/kokoro-v1_1-zh.pth -o kokoro-v1.1-zh.onnx --fp16 --int8 -v zf_001

--fp16 and --int8 write kokoro-*.fp16.onnx and kokoro-*.int8.onnx beside the
full precision file. Both keep float inputs and outputs, so they are drop in.

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


def modernize(checkpoint: Path) -> Path:
    """Rename new style weight norm keys to what KModel expects.

    Checkpoints trained on recent torch store weight norm as
    parametrizations.weight.original0/1, while KModel registers the old
    weight_g/weight_v pair. It loads such a file with strict=False, leaving
    those layers randomly initialized, and the model then emits static.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    renames = {
        ".parametrizations.weight.original0": ".weight_g",
        ".parametrizations.weight.original1": ".weight_v",
    }
    if not any(
        old in key for part in state.values() for key in part for old in renames
    ):
        return checkpoint

    for part in state.values():
        for key in list(part):
            new = key
            for old, replacement in renames.items():
                new = new.replace(old, replacement)
            if new != key:
                part[new] = part.pop(key)

    target = checkpoint.with_suffix(".weightnorm.pth")
    torch.save(state, target)
    print(f"rewrote weight norm keys of {checkpoint.name} into {target.name}")
    return target


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


def verify(model: KModelForONNX, path: Path, voice: str, strict: bool = True) -> None:
    """Check the graph against the torch model it was traced from.

    Quantized copies predict slightly different durations, so only the full
    precision export is held to identical output.
    """
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
    shared = min(len(waveform), len(reference))
    error = (
        np.abs(waveform[:shared] - reference[:shared]).max() / np.abs(reference).max()
    )
    drift = (len(waveform) - len(reference)) / SAMPLE_RATE

    print(f"  waveform  {len(waveform)} samples, {len(waveform) / SAMPLE_RATE:.2f}s")
    print(
        f"  duration  {frames} frames, {len(waveform) / frames:.0f} samples per frame"
    )
    print(
        f"  against torch  peak error {error:.1%}, correlation "
        f"{np.corrcoef(waveform[:shared], reference[:shared])[0, 1]:.4f}, "
        f"length {drift:+.2f}s"
    )
    assert len(waveform) == frames * FRAME, f"expected {FRAME} samples per frame"
    if strict:
        assert np.array_equal(duration, expected_duration.numpy()), "durations disagree"


def quantize(path: Path, fp16: bool, int8: bool) -> list[Path]:
    """Write reduced precision copies beside the full precision export."""
    written = []

    if fp16:
        # onnxconverter_common leaves this graph with mismatched Cast types,
        # onnxruntime's own converter handles them and the Loop subgraph
        from onnxruntime.transformers.onnx_model import OnnxModel

        target = path.with_suffix(".fp16.onnx")
        graph = OnnxModel(onnx.load(str(path)))
        # keep_io_types leaves inputs and outputs float32, so every variant
        # is called exactly the same way
        graph.convert_float_to_float16(
            keep_io_types=True, use_symbolic_shape_infer=False
        )
        graph.save_model_to_file(str(target))
        written.append(target)

    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        target = path.with_suffix(".int8.onnx")
        quantize_dynamic(str(path), str(target), weight_type=QuantType.QInt8)
        written.append(target)

    for target in written:
        print(f"quantized {target} ({target.stat().st_size / 1e6:.0f} MB)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Kokoro model to ONNX")
    parser.add_argument("-c", "--config", default="checkpoints/config.json")
    parser.add_argument("-p", "--checkpoint", default="checkpoints/kokoro-v1_0.pth")
    parser.add_argument("-o", "--output", default="kokoro.onnx")
    parser.add_argument(
        "-v", "--voice", default="af_heart", help="voice used to verify"
    )
    parser.add_argument("--fp16", action="store_true", help="also write a fp16 copy")
    parser.add_argument("--int8", action="store_true", help="also write an int8 copy")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = modernize(Path(args.checkpoint))
    model = KModelForONNX(
        KModel(config=args.config, model=str(checkpoint), disable_complex=True)
    ).eval()

    export(model, output, args.voice)
    embed_config(output, Path(args.config))
    variants = quantize(output, args.fp16, args.int8)

    if not args.skip_verify:
        for path in [output, *variants]:
            print(path.name)
            verify(model, path, args.voice, strict=path == output)


if __name__ == "__main__":
    main()
