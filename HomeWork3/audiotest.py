from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    # TODO: ASR → LLM → TTS
    return FileResponse("response.wav", media_type="audio/wav")


