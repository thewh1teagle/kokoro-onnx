"""Session-input dtypes fed to the ONNX model.

Regression tests for #211: with the 0.4.7 code, a model using the newer
``input_ids`` input format got ``speed`` as ``np.int32`` while the graph
expects ``tensor(float)``, so onnxruntime raised ``InvalidArgument``. The
inputs are now built from the dtypes the model itself declares
(``kokoro_onnx.session.input_dtypes``), and these tests lock that in.

The session is faked so the tests run without a model file or espeak: only
``Kokoro._infer`` and ``input_dtypes`` are exercised, which is exactly the
code that built the wrong feed.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from kokoro_onnx import Kokoro
from kokoro_onnx.session import input_dtypes


class FakeSession:
    """Stand-in for ``rt.InferenceSession``: declares inputs, captures feeds."""

    def __init__(self, inputs, outputs=("waveform", "duration")):
        self._inputs = [
            SimpleNamespace(name=name, type=type_) for name, type_ in inputs
        ]
        self._outputs = [SimpleNamespace(name=name) for name in outputs]
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={})

    def run(self, _output_names, inputs):
        self.feeds.append(inputs)
        return [np.zeros(240, dtype=np.float32), np.zeros(3, dtype=np.int64)]


def kokoro_for(session):
    """Build a ``Kokoro`` around a session, as ``_setup`` does."""
    kokoro = Kokoro.__new__(Kokoro)
    kokoro.sess = session
    kokoro._input_dtypes = input_dtypes(session)
    kokoro._tokens_input = (
        "input_ids" if "input_ids" in kokoro._input_dtypes else "tokens"
    )
    kokoro.has_timings = "duration" in {o.name for o in session.get_outputs()}
    return kokoro


FLOAT_SPEED_INPUTS = [
    ("input_ids", "tensor(int64)"),
    ("style", "tensor(float)"),
    ("speed", "tensor(float)"),
]

TOKENS_INPUTS = [
    ("tokens", "tensor(int64)"),
    ("style", "tensor(float)"),
    ("speed", "tensor(float)"),
]

INT_SPEED_INPUTS = [
    ("input_ids", "tensor(int64)"),
    ("style", "tensor(float)"),
    ("speed", "tensor(int32)"),
]


class TestInferInputs(unittest.TestCase):
    def test_input_ids_model_with_float_speed_feeds_float32(self):
        """#211: speed must be float32 for input_ids models expecting float."""
        session = FakeSession(FLOAT_SPEED_INPUTS)
        kokoro = kokoro_for(session)
        style = np.zeros(256, dtype=np.float32)

        kokoro._infer([10, 20, 30], style, 1.25)

        (feed,) = session.feeds
        self.assertEqual(feed["speed"].dtype, np.float32)
        self.assertEqual(feed["speed"].tolist(), [1.25])
        self.assertIn("input_ids", feed)
        self.assertNotIn("tokens", feed)
        self.assertEqual(feed["input_ids"].dtype, np.int64)

    def test_tokens_model_with_float_speed_feeds_float32(self):
        """The older ``tokens`` format keeps feeding float32 speed."""
        session = FakeSession(TOKENS_INPUTS)
        kokoro = kokoro_for(session)
        style = np.zeros(256, dtype=np.float32)

        kokoro._infer([10, 20, 30], style, 1.25)

        (feed,) = session.feeds
        self.assertEqual(feed["speed"].dtype, np.float32)
        self.assertIn("tokens", feed)
        self.assertNotIn("input_ids", feed)

    def test_integer_speed_model_rounds_and_feeds_int32(self):
        """Exports with an int speed input (v1.1-zh) still get int32."""
        session = FakeSession(INT_SPEED_INPUTS)
        kokoro = kokoro_for(session)
        style = np.zeros(256, dtype=np.float32)

        kokoro._infer([10, 20, 30], style, 1.5)

        (feed,) = session.feeds
        self.assertEqual(feed["speed"].dtype, np.int32)
        self.assertEqual(feed["speed"].tolist(), [2])

    def test_input_dtypes_maps_declared_types(self):
        session = FakeSession(FLOAT_SPEED_INPUTS)
        self.assertEqual(
            input_dtypes(session),
            {
                "input_ids": np.dtype(np.int64),
                "style": np.dtype(np.float32),
                "speed": np.dtype(np.float32),
            },
        )


if __name__ == "__main__":
    unittest.main()
