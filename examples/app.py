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
Run this with:
uv run examples/app.py
"""

import gradio as gr
from kokoro_onnx import Kokoro, SUPPORTED_LANGUAGES
from kokoro_onnx.tokenizer import Tokenizer


tokenizer = Tokenizer()
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")


def speak(text: str, voice: str, language: str):
    phonemes = tokenizer.phonemize(text, lang=language)
    samples, sample_rate = kokoro.create(
        text="",
        phonemes=phonemes,
        voice=voice,
        speed=1.0,
    )
    return [(sample_rate, samples), phonemes]


def create():
    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto")])) as ui:
        text_input = gr.TextArea(
            label="Input Text",
            rtl=False,
            value="Hello world!",
        )
        language_input = gr.Dropdown(
            label="Language",
            value="en-us",
            choices=SUPPORTED_LANGUAGES,
        )
        voice_input = gr.Dropdown(
            label="Voice", value="af_sarah", choices=sorted(kokoro.get_voices())
        )
        submit_button = gr.Button("Create")
        phonemes_output = gr.Textbox(label="Phonemes")
        audio_output = gr.Audio()
        submit_button.click(
            fn=speak,
            inputs=[text_input, voice_input, language_input],
            outputs=[audio_output, phonemes_output],
        )
    return ui


ui = create()
ui.launch(debug=True)
