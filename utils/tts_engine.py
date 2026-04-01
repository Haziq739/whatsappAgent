import edge_tts
import asyncio

# Voice names for Urdu and English
URDU_VOICE = "ur-PK-AsadNeural" 
ENGLISH_VOICE = "en-US-GuyNeural"

async def text_to_speech(text, lang="en"):
    voice = URDU_VOICE if lang == "ur" else ENGLISH_VOICE
    output_path = "response_audio.mp3"
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

def generate_audio_sync(text, lang="en"):
    """Helper to run the async function in your normal code"""
    return asyncio.run(text_to_speech(text, lang))