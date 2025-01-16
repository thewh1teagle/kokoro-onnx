import os
import gradio as gr  # TODO:  pip install gradio==5.12.0
import datetime
import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")

def action(infer_text, lang):
    try:
        print(f"Received infer_text: {infer_text} with lang: {lang}")
        samples, sample_rate = kokoro.create(
            infer_text, voice="af_sarah", speed=1.0, lang=lang
        )
        infer_audio_path = f"./audio_data/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        if not os.path.exists("./audio_data"):
            os.makedirs("./audio_data")
        sf.write(infer_audio_path, samples, sample_rate)
        print("Created audio.wav")
        return infer_audio_path
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

interface = gr.Interface(
    fn=action,
    inputs=[
        gr.Textbox(label="Infer Text", value="Hello, World!"),
        gr.Dropdown(label="Language", choices=["en-us", "en-gb", "fr-fr", "ja", "ko", "cmn"], value="en-us")
    ],
    outputs=gr.Audio(label="Generated Audio Files"),
    live=False
)

interface.launch(server_name="127.0.0.1", server_port=7777, share=True)
