# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio>=5.13.1",
#     "kokoro-onnx>=0.3.8",
# ]
#
# [tool.uv.sources]
# kokoro-onnx = { path = "../" }
# ///

"""
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
uv run examples/app.py
"""

import gradio as gr
from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer
import numpy as np

tokenizer = Tokenizer()
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")


SUPPORTED_LANGUAGES = ["en-us"]


def create(text: str, voice: str, language: str, blend_voice_name: str = None):
    phonemes = tokenizer.phonemize(text, lang=language)

    # Blending
    if blend_voice_name:
        first_voice = kokoro.get_voice_style(voice)
        second_voice = kokoro.get_voice_style(blend_voice_name)
        voice = np.add(first_voice * (50 / 100), second_voice * (50 / 100))
    samples, sample_rate = kokoro.create(
        phonemes, voice=voice, speed=1.0, is_phonemes=True
    )
    return [(sample_rate, samples), phonemes]


def create_app():
    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto")])) as ui:
        text_input = gr.TextArea(
            label="Input Text",
            rtl=False,
            value="Kokoro TTS. Turning words into emotion, one voice at a time!",
        )
        language_input = gr.Dropdown(
            label="Language",
            value="en-us",
            choices=SUPPORTED_LANGUAGES,
        )
        voice_input = gr.Dropdown(
            label="Voice", value="af_sky", choices=sorted(kokoro.get_voices())
        )
        blend_voice_input = gr.Dropdown(
            label="Blend Voice (Optional)",
            value=None,
            choices=sorted(kokoro.get_voices()) + [None],
        )
        submit_button = gr.Button("Create")
        phonemes_output = gr.Textbox(label="Phonemes")
        audio_output = gr.Audio()
        submit_button.click(
            fn=create,
            inputs=[text_input, voice_input, language_input, blend_voice_input],
            outputs=[audio_output, phonemes_output],
        )
    return ui


ui = create_app()
ui.launch(debug=True)
