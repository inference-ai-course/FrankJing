import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from transformers import pipeline
import edge_tts

app = FastAPI()
# tts_engine = CozyVoice()
asr_model = whisper.load_model("small")
llm = pipeline("text-generation", model="meta-llama/Llama-3-8B")
conversation_history=[]


def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result["text"]

# def synthesize_speech(text, filename="response.wav"):
#     tts_engine.generate(text, output_file=filename)
#     return filename
async def synthesize_speech(text: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(output_path)
    return output_path

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)
    bot_text = generate_response(user_text)
    audio_path = synthesize_speech(bot_text)
    return FileResponse(audio_path, media_type="audio/wav")







