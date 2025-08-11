import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from transformers import pipeline
from TTS.api import TTS  # Coqui TTS
import torch
import uvicorn
from openai import OpenAI

app = FastAPI()

# Load ASR model
asr_model = whisper.load_model("small")

# Load LLM
#llm = pipeline("text-generation", model="meta-llama/Llama-3-8B")
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

# Load Coqui TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

conversation_history = []


def transcribe_audio(audio_bytes):
    """Convert speech to text using Whisper."""
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result["text"]


def synthesize_speech(text: str, output_path: str):
    """Generate speech audio from text using Coqui TTS."""
    tts_model.tts_to_file(text=text, file_path=output_path)
    return output_path


def generate_response(user_text):
    """Generate an assistant's reply using the LLM."""
    conversation_history.append({"role": "user", "text": user_text})
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    response = client.chat.completions.create(model="llama2",messages=prompt)
    bot_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)
    bot_text = generate_response(user_text)
    audio_path = "response.wav"
    synthesize_speech(bot_text, audio_path)
    return FileResponse(audio_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)