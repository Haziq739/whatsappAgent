from transformers import pipeline
import torch
from config import WHISPER_MODEL

# Initialize Whisper
stt_model = pipeline(
    "automatic-speech-recognition", 
    model=WHISPER_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def transcribe_audio(audio_path):
    result = stt_model(audio_path)
    return result["text"].strip()