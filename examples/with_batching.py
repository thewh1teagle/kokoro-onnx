"""
pip install soundfile onnxruntime numpy pydub kokoro_onnx

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python examples/with_batching.py
"""

import soundfile as sf
import sys
import time
import os
from kokoro_onnx import Kokoro
import onnxruntime as ort
import numpy as np
import re
from pydub import AudioSegment

# onnxruntime disable warnings
ort.set_default_logger_severity(3)

# enable/disable logging of kokoro_onnx
# logging.getLogger(kokoro_onnx.__name__).setLevel("DEBUG")

# select batching size
MAX_WORDS_PER_BATCH = 21000

voices = {
    1: ("en-us_female", {1: "af_alloy", 2: "af_aoede", 3: "af_bella", 4: "af_heart", 5: "af_jessica", 6: "af_kore", 7: "af_nicole", 8: "af_nova", 9: "af_river", 10: "af_sarah", 11: "af_sky"}),
    2: ("en-us_male", {1: "am_adam", 2: "am_echo", 3: "am_eric", 4: "am_fenrir", 5: "am_liam", 6: "am_michael", 7: "am_onyx", 8: "am_puck"}),
    3: ("en-gb", {1: "bf_alice", 2: "bf_emma", 3: "bf_isabella", 4: "bf_lily", 5: "bm_daniel", 6: "bm_fable", 7: "bm_george", 8: "bm_lewis"})
}

def count_words(text):
    """
    Counts the number of words in a given text.

    :param text: Input text.
    :return: Number of words.
    """
    return len(text.split())

def display_process_statistics(text):
    """
    Displays process statistics (total words, estimated batches, estimated generation time)
    and asks the user if they want to continue.
    
    :param text: Input text.
    """
    total_words = count_words(text)
    estimated_batches = (total_words // MAX_WORDS_PER_BATCH) + (1 if total_words % MAX_WORDS_PER_BATCH else 0)
    estimated_time = estimate_generation_time(total_words)
    
    print("\nPROCESS STATISTICS:")
    print(f"Total words: {total_words}")
    print(f"Estimated batches: {estimated_batches}")
    print(f"Estimated generation time: {estimated_time:.2f} seconds ({estimated_time/60:.2f} minutes, {estimated_time/3600:.2f} hours)")
    
    confirm = input("Do you want to continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Process aborted.")
        sys.exit(0)

def split_text_into_batches(text, max_words_per_batch=MAX_WORDS_PER_BATCH):
    """
    Splits a given text into batches of approximately equal size.
    
    :param text: Input text.
    :param max_words_per_batch: Maximum number of words per batch.
    :return: List of batches, where each batch is a string of sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    batches = []
    current_batch = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = count_words(sentence)
        if current_word_count + word_count > max_words_per_batch:
            batches.append(' '.join(current_batch))
            current_batch = []
            current_word_count = 0
        current_batch.append(sentence)
        current_word_count += word_count
    
    if current_batch:
        batches.append(' '.join(current_batch))
    
    return batches

def estimate_generation_time(word_count, base_words=20000, base_time=180):
    """
    Estimates the generation time for a given word count.
    
    :param word_count: Total number of words in the document.
    :param base_words: Base number of words used for estimation (default is 20000).
    :param base_time: Base time used for estimation (default is 180 seconds).
    :return: Estimated generation time in seconds.
    """
    return (word_count / base_words) * base_time

def concat_audio_parts(output_dir, base_name, selected_voice):
    """
    Concatenates multiple WAV files into a single WAV file.
    
    :param output_dir: Directory where the WAV files are stored.
    :param base_name: Base name of the generated files.
    :param selected_voice: Selected voice identifier to match file naming.
    """
    
    part_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(f"{base_name}_read_by_{selected_voice[3:]}_part_") and f.endswith(".wav")],
        key=lambda x: int(x.split("_part_")[-1].split(".wav")[0])
    )
    
    if not part_files:
        print("No audio parts found to concatenate.")
        return
    
    audio_data = []
    sample_rate = None
    
    for part in part_files:
        samples, sr = sf.read(os.path.join(output_dir, part))
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            print(f"Warning: Sample rate mismatch in {part}, skipping.")
            continue
        audio_data.append(samples)
    
    full_audio = np.concatenate(audio_data, axis=0)
    final_output_filename = os.path.join(output_dir, f"{base_name}_read_by_{selected_voice[3:]}_FULL.wav")
    sf.write(final_output_filename, full_audio, sample_rate)
    print(f"Successfully created concatenated file: {final_output_filename}")

def select_voice():
    """
    Prompts the user to select a voice category and voice identifier.
    
    :return: The selected voice identifier (e.g. "af_sky").
    """
    print("Available voice categories:")
    for num, (category, _) in voices.items():
        print(f"{num}. {category}")
    
    selected_category = int(input("Enter the number of the category: "))
    category_name, available_voices = voices.get(selected_category, voices[1])
    
    print(f"Available voices in {category_name}:")
    for num, voice in available_voices.items():
        print(f"{num}. {voice}")
    
    selected_voice_num = int(input("Enter the number of the voice: "))
    return available_voices.get(selected_voice_num, "af_sky")


def convert_audio(input_file, output_format="mp3"):
    """
    Converts an audio file from WAV to the specified format.

    :param input_file: Path to the input WAV file.
    :param output_format: The desired output format (default is "mp3").
    :return: The path to the converted audio file if successful, None otherwise.
    """
    try:
        audio = AudioSegment.from_wav(input_file)
        output_file = input_file.replace(".wav", f".{output_format}")
        
        if not os.path.exists("mp3"):
            os.makedirs("mp3")
            
        output_file = os.path.join("mp3", os.path.basename(output_file))
        audio.export(output_file, format=output_format)
        print(f"Successfully converted {input_file} to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None


def generate_audio_batches(text, output_dir, base_name, selected_voice):
    """
    Generates audio files from text by processing it in batches.

    This function splits the input text into batches, generates audio for each batch
    using a specified voice, and saves the audio files to the specified output directory.
    It also calculates and displays processing statistics and execution time for each batch.

    :param text: The input text to be converted into audio.
    :param output_dir: The directory where the generated audio files will be saved.
    :param base_name: The base name for the output audio files.
    :param selected_voice: The voice identifier to be used for audio generation.
    """

    display_process_statistics(text)
    batches = split_text_into_batches(text)
    start_time = time.time()
    
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    
    for i, batch in enumerate(batches):
        try:
            words_count = count_words(batch)
            est_time = estimate_generation_time(words_count)
            print("-------------------------------------------------------------")
            print(f"{i+1}. Processing batch {i+1}/{len(batches)}")
            print(f"Generating audio with voice: {selected_voice}")
            print(f"Text length: {len(batch)} characters")
            print(f"Words count: {words_count}")
            print(f"Estimated generation time: {est_time:.2f} seconds, {est_time/60:.2f} minutes, {est_time/3600:.2f} hours")
            
            output_filename = os.path.join(output_dir, f"{base_name}_read_by_{selected_voice[3:]}_part_{i+1}.wav")
            samples, sample_rate = kokoro.create(batch, voice=selected_voice, speed=1.0, lang="en-us")
            sf.write(output_filename, samples, sample_rate)
            
            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Execution time for batch {i+1}: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        except Exception as e:
            print(f"Error generating batch {i+1}: {e}")
            continue
    
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("------------------------------------------------------------")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    try:
        concat_audio_parts(output_dir, base_name, selected_voice)
        convert_audio(os.path.join(output_dir, f"{base_name}_read_by_{selected_voice[3:]}_FULL.wav"), "mp3")
    except Exception as e:
        print(f"Error concatenating audio parts: {e}")

# Function to generate sample audio files for each voice
def demo_voices():
    """
    Generates a sample audio file for each voice in the voices dictionary.

    This function generates audio using the kokoro model for each voice in the voices dictionary.
    The generated audio files are saved in subfolders corresponding to the category/locale of
    each voice.

    The function takes no arguments and does not return any value.
    """
    base_output_dir = "demo_voices"
    # Потребителски избор на текст
    input_text = "This is a demo text to test the voice "
    os.makedirs(base_output_dir, exist_ok=True)

    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    for category_num, (category_name, available_voices) in voices.items():
        # Create a subfolder for each locale/category
        category_dir = os.path.join(base_output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)

        for voice_num, voice in available_voices.items():
            try:
                print(f"Generating audio for voice: {voice} in category: {category_name}")
                samples, sample_rate = kokoro.create(
                    input_text + voice[3:], voice=voice, speed=1.0, lang="en-us"
                )

                # Save audio to the respective subfolder
                output_filename = os.path.join(category_dir, f"{voice}.wav")
                sf.write(output_filename, samples, sample_rate)
                print(f"Created {output_filename}")
            except Exception as e:
                print(f"Error generating audio for voice {voice}: {e}")


def main():
    """
    Main entry point for the script.

    This function parses the command line arguments and processes the following steps:
    1. Checks if the input file exists.
    2. Reads the text from the input file.
    3. Asks the user to select a voice.
    4. Generates audio for the given text using the selected voice.

    :return: None
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_text_file>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    if input_filename == "demo_voices":
        demo_voices()
        sys.exit(0)

    if not os.path.isfile(input_filename):
        print(f"Error: File '{input_filename}' does not exist.")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_dir = os.path.join("wav", base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_filename, "r", encoding="utf-8") as file:
        text = file.read().strip()
    
    selected_voice = select_voice()
    generate_audio_batches(text, output_dir, base_name, selected_voice)

if __name__ == "__main__":
    main()
