from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import os
from collections import deque
import openai

# Initialize FastAPI
app = FastAPI()

# Configure API key
openai.api_key = os.getenv("OPENAI")

# Memory: store last 5 turns [(user_text, bot_text), ...]
conversation_history = deque(maxlen=5)

# Helper: Speech-to-Text (ASR)
def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # or "whisper-1"
            file=audio_file
        )
    return transcript.text

# Helper: Generate LLM Response with Memory
def generate_llm_response(user_input: str) -> str:
    messages = []
    for u, b in conversation_history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_input})

    completion = openai.Chat.completions.create(
        model="gpt-4o-mini",  # Low-latency LLM
        messages=messages
    )
    return completion.choices[0].message.content

# Helper: Text-to-Speech (TTS)
def synthesize_speech(text: str, output_path: str):
    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(output_path)

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        tmp_input.write(await file.read())
        tmp_input_path = tmp_input.name

    # Step 1: ASR
    user_text = transcribe_audio(tmp_input_path)

    # Step 2: LLM
    bot_text = generate_llm_response(user_text)

    # Update memory
    conversation_history.append((user_text, bot_text))

    # Step 3: TTS
    tmp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    synthesize_speech(bot_text, tmp_output_path)

    # Cleanup input audio
    os.unlink(tmp_input_path)

    # Return audio file
    return FileResponse(tmp_output_path, media_type="audio/wav", filename="response.wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
