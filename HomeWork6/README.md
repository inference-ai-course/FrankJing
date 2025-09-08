# Voice Agent with Function Calling - Week 6 Assignment

This project extends the Week 3 voice agent with function calling capabilities, allowing the AI to automatically execute tools based on natural language commands.

## Features

- **Speech-to-Text**: Uses OpenAI Whisper for voice transcription
- **Function Calling**: LLM can automatically call functions for:
  - Mathematical calculations using SymPy
  - ArXiv paper searches
- **Text-to-Speech**: Converts responses to speech using gTTS
- **Multi-turn Conversations**: Maintains conversation history
- **Web Interface**: Simple HTML interface for testing

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama Server** (required for local LLM):
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

3. **Run the Voice Agent**:
   ```bash
   python voice_agent.py
   ```

4. **Access the Web Interface**:
   Open your browser and go to `http://127.0.0.1:8000`

## Usage

The voice agent can handle three types of queries:

### 1. Mathematical Calculations
- **Example**: "What is 15 times 23?"
- **Response**: The agent will use the calculate function and return the result

### 2. ArXiv Paper Searches  
- **Example**: "Find papers about neural networks"
- **Response**: The agent will search arXiv and return relevant papers

### 3. General Conversation
- **Example**: "Hello, how are you?"
- **Response**: Normal conversational response without function calling

## Function Calling Architecture

The system works as follows:

1. **User speaks** → Whisper transcribes to text
2. **Text sent to LLM** (Llama 3.2) with function calling instructions
3. **LLM decides** whether to make a function call or respond normally
4. **If function call**: JSON output is parsed and appropriate function executed
5. **Result converted to speech** using gTTS and returned to user

### Function Call Format

The LLM outputs JSON when a function call is needed:

```json
{"function": "calculate", "arguments": {"expression": "2+2"}}
```

```json
{"function": "search_arxiv", "arguments": {"query": "quantum computing"}}
```

## Testing

Run the test script to verify function calling without the full voice pipeline:

```bash
python test_functions.py
```

This will test:
- Mathematical calculations
- ArXiv searches  
- Function routing logic
- Generate sample logs as required by the assignment

## Project Structure

```
HomeWork6/
├── voice_agent.py          # Main voice agent with function calling
├── test_functions.py       # Test script for function calling
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── CLAUDE.md              # Assignment instructions
```

## Sample Test Logs

### Test 1: Math Query
1. User query: "What is 15 times 23?"
2. Raw LLM response: `{"function": "calculate", "arguments": {"expression": "15*23"}}`
3. Function call made: `calculate("15*23")`
4. Final response: "The result of 15*23 is: 345"

### Test 2: ArXiv Search
1. User query: "Find papers about quantum computing"
2. Raw LLM response: `{"function": "search_arxiv", "arguments": {"query": "quantum computing"}}`
3. Function call made: `search_arxiv("quantum computing")`
4. Final response: [ArXiv search results with paper titles, authors, and summaries]

### Test 3: Normal Conversation
1. User query: "Hello, how are you?"
2. Raw LLM response: "Hello! I'm doing well, thank you for asking..."
3. Function call made: None (regular text response)
4. Final response: "Hello! I'm doing well, thank you for asking..."

## Error Handling

The system gracefully handles:
- Invalid mathematical expressions
- ArXiv API failures (falls back to dummy responses)
- Malformed JSON from LLM
- Unknown function calls
- Audio processing errors

## Dependencies

- **FastAPI**: Web framework for API endpoints
- **Whisper**: Speech-to-text transcription
- **OpenAI**: LLM client for Ollama
- **gTTS**: Text-to-speech synthesis
- **SymPy**: Mathematical expression evaluation
- **arxiv**: ArXiv API client for paper searches
- **transformers & torch**: ML model dependencies