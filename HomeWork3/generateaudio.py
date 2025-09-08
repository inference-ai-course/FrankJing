import asyncio
# from TTS.api import TTS 
from gtts import gTTS
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_audio_path = os.path.join(BASE_DIR, "temp_g.wav")
device = "cuda" if torch.cuda.is_available() else "cpu"
# tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

# def generate_audio(text, filename="temp.wav"):
#     tts_model.tts_to_file(text=text, file_path=input_audio_path)

text = "Will IT professionals lost job because of Artificial Intelligence ?"
# generate_audio(text)
tts = gTTS(text=text, lang="en")
tts.save(input_audio_path)
