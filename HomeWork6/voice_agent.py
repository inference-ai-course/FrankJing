import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import OpenAI
import tempfile
import os
from gtts import gTTS
from function_tools import create_agent
from langchain_openai import ChatOpenAI

app = FastAPI()

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ASR (Whisper) model
# -----------------------------
asr_model = whisper.load_model("small")

# -----------------------------
# Initialize LangGraph Agent with Ollama
# -----------------------------
# Use Ollama through OpenAI-compatible API
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3.2",
    temperature=0.7
)

# Create the function calling agent
agent = create_agent(llm)


# Function definitions now handled by LangGraph agent in function_tools.py


def transcribe_audio(audio_bytes: bytes) -> str:
    """Convert speech to text using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name
    try:
        result = asr_model.transcribe(temp_path)
    finally:
        os.remove(temp_path)
    return result["text"]


def synthesize_speech(text: str, output_path: str) -> str:
    """Generate speech audio from text using gTTS."""
    tts_model = gTTS(text=text, lang="en")
    tts_model.save(output_path)
    return output_path


def generate_response(user_text: str, thread_id: str = "default") -> str:
    """Generate an assistant's reply using the LangGraph agent."""
    try:
        # Use the LangGraph agent to process the query
        response = agent.process_query(user_text, thread_id)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    """Main chat endpoint that handles voice input and returns voice output."""
    try:
        # Speech-to-text
        audio_bytes = await file.read()
        user_text = transcribe_audio(audio_bytes)
        print(f"User said: {user_text}")

        # LLM response with function calling
        bot_text = generate_response(user_text)
        print(f"Bot response: {bot_text}")

        # Text-to-speech
        audio_path = "response.wav"
        synthesize_speech(bot_text, audio_path)

        return FileResponse(audio_path, media_type="audio/wav")
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        # Return error message as audio
        error_text = "Sorry, I encountered an error processing your request."
        audio_path = "error_response.wav"
        synthesize_speech(error_text, audio_path)
        return FileResponse(audio_path, media_type="audio/wav")


@app.post("/chat/text/")
async def text_chat_endpoint(request: dict):
    """Text-based chat endpoint for React frontend."""
    try:
        user_text = request.get("message", "")
        thread_id = request.get("thread_id", "default")
        
        if not user_text:
            return {"error": "No message provided"}
        
        print(f"User message: {user_text}")
        
        # LLM response with function calling
        bot_text = generate_response(user_text, thread_id)
        print(f"Bot response: {bot_text}")
        
        return {
            "response": bot_text,
            "thread_id": thread_id
        }
    
    except Exception as e:
        print(f"Error in text chat endpoint: {e}")
        return {
            "error": f"Sorry, I encountered an error: {str(e)}"
        }


@app.get("/")
def home():
    """Serve the web interface for testing."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Agent with Function Calling</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            button { padding: 10px 20px; margin: 10px; font-size: 16px; }
            .status { margin: 20px 0; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice Agent with Function Calling</h1>
            <p>This voice agent can:</p>
            <ul>
                <li>Answer general questions</li>
                <li>Perform mathematical calculations</li>
                <li>Search arXiv papers</li>
            </ul>
            
            <h2>Upload Audio File</h2>
            <form action="/chat/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="audio/*" required>
                <br><br>
                <input type="submit" value="Send Audio">
            </form>
            
            <div class="status">
                <h3>Try asking:</h3>
                <ul>
                    <li>"What is 15 times 23?"</li>
                    <li>"Find papers about neural networks"</li>
                    <li>"Hello, how are you today?"</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)