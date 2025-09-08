import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from transformers import pipeline
#from TTS.api import TTS  # Coqui TTS
import torch
import uvicorn
from openai import OpenAI
import tempfile
import os,asyncio
from gtts import gTTS
from fastapi.responses import HTMLResponse
# import edge_tts


app = FastAPI()

# -----------------------------
# Load ASR (Whisper) model
# -----------------------------
asr_model = whisper.load_model("small")

# -----------------------------
# Load LLM client (Ollama local API)
# -----------------------------
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # required but unused for Ollama
)

# client = OpenAI(api_key=os.getenv("OPENAI"))

# -----------------------------
# Load Coqui TTS model
# -----------------------------
# tts_model = TTS(
#     model_name="tts_models/en/ljspeech/tacotron2-DDC",
#     progress_bar=False
# )
# Conversation memory
conversation_history = []


def transcribe_audio(audio_bytes: bytes) -> str:
    """Convert speech to text using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name
    try:
        result = asr_model.transcribe(temp_path)
    finally:
        os.remove(temp_path)  # cleanup temp file
    return result["text"]


def synthesize_speech(text: str, output_path: str) -> str:
    """Generate speech audio from text using Coqui TTS."""
    # tts_model.tts_to_file(text=text, file_path=output_path)
    # async def _speak():
    #     tts = edge_tts.Communicate(text, "en-US-AriaNeural")
    #     await tts.save(output_path)

    # asyncio.run(_speak())
    # return output_path
    
    tts_model=gTTS(text=text, lang="en")
    tts_model.save(output_path)
    return output_path


def generate_response(user_text: str) -> str:
    """Generate an assistant's reply using the LLM."""
    # Append user turn
    conversation_history.append({"role": "user", "text": user_text})

    # Convert last 5 turns to OpenAI chat format
    messages = [
        {"role": turn["role"], "content": turn["text"]}
        for turn in conversation_history[-5:]
    ]

    #Get model response
    response = client.chat.completions.create(
        model="llama2",
        messages=messages
    )
    bot_response = response.choices[0].message.content

    # Get model response
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",  # replace "llama2" with a valid OpenAI model
    #     messages=messages
    # )
    # # Extract assistant message
    # bot_response = response.choices[0].message.content
    # Append assistant turn
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    # Speech-to-text
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)

    # LLM response
    bot_text = generate_response(user_text)

    # Text-to-speech
    audio_path = "response.wav"
    synthesize_speech(bot_text, audio_path)

    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/")
def home():
    return HTMLResponse("""
    <html>
    <body>
        <h2>Upload Audio</h2>
        <form action="/chat/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
        </form>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
