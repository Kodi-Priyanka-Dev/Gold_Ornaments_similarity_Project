import os
from faster_whisper import WhisperModel

# Initialize the Whisper model
# We use the "small" model as requested. It is accurate enough for search queries and fast.
# Using CPU by default, adjust to "cuda" if GPU is available.
model_size = "small"

print(f"Loading faster-whisper ({model_size}) model into memory...")
try:
    # Use compute_type="int8" for CPU, "float16" for GPU
    device = "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("faster-whisper model loaded successfully.")
except Exception as e:
    print(f"Failed to load faster-whisper model: {e}")
    whisper_model = None

def transcribe_audio(audio_path, language_hint=None):
    """
    Transcribes the given audio file using faster-whisper.
    :param audio_path: Path to the audio file.
    :param language_hint: 2-letter language code hint (e.g. 'te' for Telugu)
    :return: Transcribed text string
    """
    if not whisper_model:
        raise RuntimeError("Whisper model is not initialized.")
        
    print(f"Transcribing {audio_path} (Hint: {language_hint})...")
    
    # We can pass language hint if available
    segments, info = whisper_model.transcribe(audio_path, beam_size=5, language=language_hint)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability}")
    
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)
        
    full_text = " ".join(text_parts).strip()
    return full_text
