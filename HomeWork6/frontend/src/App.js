import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Bot, User, Calculator, BookOpen, MessageCircle } from 'lucide-react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your AI assistant. I can help you with general questions, perform mathematical calculations, and search academic papers on arXiv. How can I assist you today?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [threadId] = useState('default-' + Date.now());
  
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendTextMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat/text/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          thread_id: threadId
        })
      });

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.error || data.response,
        timestamp: new Date(),
        isError: !!data.error
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error connecting to the server. Please try again.',
        timestamp: new Date(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await sendAudioMessage(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioMessage = async (audioBlob) => {
    // Add user message indicating voice input was received
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: '🎤 Voice message sent',
      timestamp: new Date(),
      isVoice: true
    };
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.wav');

      const response = await fetch(`${API_BASE_URL}/chat/`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        // Get the audio response and create an audio element to play it
        const audioResponse = await response.blob();
        const audioUrl = URL.createObjectURL(audioResponse);
        
        // Create audio element and play the response
        const audio = new Audio(audioUrl);
        audio.play().catch(e => console.error('Error playing audio:', e));
        
        const botMessage = {
          id: Date.now(),
          type: 'bot',
          content: '🔊 Audio response (click to replay)',
          timestamp: new Date(),
          isVoice: true,
          audioUrl: audioUrl
        };

        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error('Failed to process audio');
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        content: 'Sorry, I encountered an error processing your audio. Please try again.',
        timestamp: new Date(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  const getMessageIcon = (type, content, isVoice) => {
    if (type === 'user') {
      return isVoice ? <Mic className="w-5 h-5" /> : <User className="w-5 h-5" />;
    }
    
    // Detect function usage based on content
    if (content.includes('The result of') || content.includes('calculate')) {
      return <Calculator className="w-5 h-5" />;
    }
    if (content.includes('Found') && content.includes('papers') || content.includes('arXiv')) {
      return <BookOpen className="w-5 h-5" />;
    }
    
    return <Bot className="w-5 h-5" />;
  };

  const handleAudioMessageClick = (audioUrl) => {
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.play().catch(e => console.error('Error playing audio:', e));
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <div className="header-icon">
              <MessageCircle className="w-6 h-6" />
            </div>
            <div className="header-text">
              <h1>AI Voice Agent</h1>
              <p>Function Calling Assistant</p>
            </div>
          </div>
        </div>

        <div className="messages-container">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
                <div className="message-icon">
                  {getMessageIcon(message.type, message.content, message.isVoice)}
                </div>
                <div className="message-body">
                  <div 
                    className={`message-text ${message.isError ? 'error' : ''} ${message.isVoice ? 'voice-message' : ''}`}
                    onClick={message.audioUrl ? () => handleAudioMessageClick(message.audioUrl) : undefined}
                    style={message.audioUrl ? { cursor: 'pointer' } : {}}
                  >
                    {message.content}
                  </div>
                  <div className="message-timestamp">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot">
              <div className="message-content">
                <div className="message-icon">
                  <Bot className="w-5 h-5" />
                </div>
                <div className="message-body">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="example-queries">
            <button 
              onClick={() => setInputMessage("What is 15 times 23?")}
              className="example-btn"
            >
              <Calculator className="w-4 h-4" />
              Math Example
            </button>
            <button 
              onClick={() => setInputMessage("Find papers about quantum computing")}
              className="example-btn"
            >
              <BookOpen className="w-4 h-4" />
              arXiv Example
            </button>
          </div>

          {/* Main Record Button */}
          <div className="record-section">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`main-record-btn ${isRecording ? 'recording' : ''}`}
              disabled={isLoading}
            >
              <div className="record-btn-content">
                {isRecording ? (
                  <>
                    <MicOff className="w-6 h-6" />
                    <span>Stop Recording</span>
                    <div className="recording-pulse"></div>
                  </>
                ) : (
                  <>
                    <Mic className="w-6 h-6" />
                    <span>Press to Record</span>
                  </>
                )}
              </div>
            </button>
            
            {isRecording && (
              <div className="recording-status">
                <div className="recording-indicator">
                  <div className="pulse-dot"></div>
                  Recording...
                </div>
              </div>
            )}
          </div>

          {/* Text Input Area */}
          <div className="text-input-section">
            <div className="input-area">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Or type your message here..."
                className="message-input"
                disabled={isLoading}
                rows="1"
              />
              
              <button
                onClick={sendTextMessage}
                className="send-btn"
                disabled={!inputMessage.trim() || isLoading}
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;