import gradio as gr
import soundfile as sf
from kokoro_onnx import Kokoro
import argparse

# Initialize Kokoro
kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")

# Voice categories
VOICES = {
    "American Female": ["af", "af_bella", "af_nicole", "af_sarah", "af_sky"],
    "American Male": ["am_adam", "am_michael"],
    "British Female": ["bf_emma", "bf_isabella"],
    "British Male": ["bm_george", "bm_lewis"]
}

def generate_audio(text, voice):
    """Generate audio from text using selected voice"""
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=1.0,
        lang="en-us"
    )
    output_path = "output.wav"
    sf.write(output_path, samples, sample_rate)
    return output_path

# Flatten voice list for dropdown
voice_choices = []
for category, voices in VOICES.items():
    for voice in voices:
        voice_choices.append(f"{category} - {voice}")

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Text-to-Speech Generator")
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Text to convert",
            placeholder="Enter your text here...",
            lines=5
        )
        
    with gr.Row():
        voice_dropdown = gr.Dropdown(
            choices=voice_choices,
            label="Select Voice",
            value=voice_choices[0]
        )
        
    with gr.Row():
        generate_btn = gr.Button("Generate Audio")
        
    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio")

    # Connect components
    generate_btn.click(
        fn=lambda text, voice: generate_audio(text, voice.split(" - ")[1]),
        inputs=[text_input, voice_dropdown],
        outputs=audio_output
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run on 0.0.0.0")
    args = parser.parse_args()
    
    if args.server:
        app.launch(server_name="0.0.0.0")
    else:
        app.launch()
