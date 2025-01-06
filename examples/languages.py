"""
Example demonstrating language support in kokoro-onnx
Based on Kokoro-TTS v0.23 which supports 5 languages:
- English (en-US, en-GB)
- French (fr-FR) 
- Japanese (ja-JP)
- Korean (ko-KR)
- Chinese (zh-CN)

Note: For CJK languages (Chinese, Japanese, Korean), English letters are not yet 
properly handled by the tokenizers. Convert or remove English text for these languages.

Usage:
    pip install kokoro-onnx soundfile
    wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
    wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
    python examples/languages.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro

# Test sentences for each supported language
sentences = {
    # English variants
    "en-us": "Hello, this is a test of US English synthesis.",
    "en-gb": "Hello, this is a test of British English synthesis.",
    
    # French
    "fr-fr": "Bonjour, ceci est un test de synthèse en français.",
    
    # Japanese (avoiding English characters)
    "ja-jp": "これは日本語の音声合成のテストです。",
    
    # Korean (avoiding English characters)  
    "ko-kr": "이것은 한국어 음성 합성 테스트입니다.",
    
    # Chinese (avoiding English characters)
    "zh-cn": "这是中文语音合成测试。",
}

def test_languages(model_path="kokoro-v0_19.onnx", voices_path="voices.json"):
    """Test TTS generation for all supported languages"""
    kokoro = Kokoro(model_path, voices_path)
    
    # Test each language with default voice
    for lang, text in sentences.items():
        try:
            print(f"\nGenerating speech for {lang}...")
            print(f"Text: {text}")
            
            # Generate speech
            samples, sample_rate = kokoro.create(
                text=text,
                voice="af",  # Using a stable voice
                speed=1.0,
                lang=lang
            )
            
            # Save the audio file
            output_file = f"test_{lang}.wav"
            sf.write(output_file, samples, sample_rate)
            print(f"✓ Successfully created {output_file}")
            
        except Exception as e:
            print(f"✗ Error generating {lang}: {str(e)}")

if __name__ == "__main__":
    test_languages()
