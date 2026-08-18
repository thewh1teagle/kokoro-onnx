"""Building and introspecting the onnxruntime session."""

import importlib.metadata
import json
import os

import numpy as np
import onnxruntime as rt

from .log import log

# Where scripts/export.py stores config.json inside the graph
CONFIG_KEY = "kokoro_config"

# Exports disagree on input types: v1.0 takes a float speed, v1.1-zh an int one.
_NUMPY_TYPES: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
}

# Distributions shipping accelerated providers. They all install the module
# `onnxruntime`, so they can only be found by distribution name.
_ACCELERATED_DISTRIBUTIONS = (
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-rocm",
    "onnxruntime-qnn",
)


def _installed(distribution: str) -> bool:
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def resolve_providers() -> list[str]:
    """Pick the execution providers to run the model with.

    ONNX_PROVIDER wins if set, otherwise every available provider is used when
    an accelerated onnxruntime distribution is installed (kokoro-onnx[gpu]),
    and plain CPU otherwise.
    """
    available = rt.get_available_providers()

    env_provider = os.getenv("ONNX_PROVIDER")
    if env_provider:
        if env_provider not in available:
            log.warning(
                f"ONNX_PROVIDER={env_provider} is not available in this onnxruntime "
                f"build, available providers: {available}"
            )
        return [env_provider]

    accelerated = [d for d in _ACCELERATED_DISTRIBUTIONS if _installed(d)]
    if accelerated:
        log.debug(f"Accelerated onnxruntime found: {accelerated}")
        return available

    return ["CPUExecutionProvider"]


def create_session(model_path: str) -> rt.InferenceSession:
    """Load the model on the providers this installation can use."""
    providers = resolve_providers()
    log.debug(f"Providers: {providers}")
    return rt.InferenceSession(model_path, providers=providers)


def embedded_vocab(session: rt.InferenceSession) -> dict:
    """The vocabulary the model was exported with, empty if it carries none."""
    config = session.get_modelmeta().custom_metadata_map.get(CONFIG_KEY)
    if not config:
        return {}
    try:
        return json.loads(config)["vocab"]
    except (ValueError, KeyError) as error:
        log.warning(f"Ignoring unreadable {CONFIG_KEY} metadata: {error}")
        return {}


def input_dtypes(session: rt.InferenceSession) -> dict[str, np.dtype]:
    """Map each model input to the numpy dtype its tensor must be built with."""
    return {
        i.name: _NUMPY_TYPES.get(i.type, np.dtype(np.float32))
        for i in session.get_inputs()
    }
