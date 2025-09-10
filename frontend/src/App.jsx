import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  // State management
  const [messages, setMessages] = useState([]);
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceConfig, setVoiceConfig] = useState({
    voiceType: 'female',
    useWhisper: true,
    voiceEnabled: true
  });
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [uiMinimized, setUiMinimized] = useState(false);
  const [inputText, setInputText] = useState('');

  // Refs
  const audioRef = useRef(null);
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (role, content, metadata = {}) => {
    const message = {
      id: Date.now() + Math.random(),
      role,
      content,
      timestamp: new Date(),
      ...metadata
    };
    setMessages(prev => [...prev, message]);
    return message;
  };

  // Simulate API call to backend
  const simulateAPICall = async (text) => {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Check for action commands
    const lowerText = text.toLowerCase();
    let actionExecuted = null;
    let response = '';

    if (lowerText.includes('open youtube')) {
      actionExecuted = 'open_youtube';
      response = "Opening YouTube for you! 🎥";
      window.open('https://www.youtube.com', '_blank');
    } else if (lowerText.includes('search for') || lowerText.includes('google')) {
      const searchQuery = text.replace(/search for|google/gi, '').trim();
      actionExecuted = 'search';
      response = `Searching for "${searchQuery}" on Google! 🔍`;
      const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`;
      window.open(searchUrl, '_blank');
    } else if (lowerText.includes('open music') || lowerText.includes('spotify')) {
      actionExecuted = 'open_music';
      response = "Opening music for you! 🎵";
      window.open('https://music.youtube.com', '_blank');
    } else if (lowerText.includes('weather')) {
      actionExecuted = 'weather';
      response = "Here's the weather information! ☀️";
      window.open('https://weather.com', '_blank');
    } else if (lowerText.includes('time')) {
      actionExecuted = 'time';
      const currentTime = new Date().toLocaleTimeString();
      response = `The current time is ${currentTime} ⏰`;
    } else {
      // General AI responses
      const responses = [
        "That's interesting! Tell me more about that.",
        "I understand what you're saying. How can I help you further?",
        "Thanks for sharing that with me! What would you like to do next?",
        "I'm here to help! Is there anything specific you'd like me to assist with?",
        "That's a great question! Let me think about that for you.",
        "I appreciate you talking with me! How else can I be of service?"
      ];
      response = responses[Math.floor(Math.random() * responses.length)];
    }

    return {
      response,
      action_executed: actionExecuted,
      audio_file: null // We'll implement TTS later
    };
  };

  // Handle text message
  const handleTextMessage = async (text) => {
    if (!text.trim()) return;

    // Add user message
    addMessage('user', text);
    setIsProcessing(true);

    try {
      const response = await simulateAPICall(text);

      // Add AI response
      addMessage('assistant', response.response, {
        actionExecuted: response.action_executed
      });

      // Speak response if voice is enabled
      if (voiceConfig.voiceEnabled && 'speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(response.response);
        utterance.voice = speechSynthesis.getVoices().find(voice => 
          voiceConfig.voiceType === 'female' ? voice.name.includes('Female') || voice.name.includes('Samantha') : voice.name.includes('Male')
        ) || speechSynthesis.getVoices()[0];
        speechSynthesis.speak(utterance);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('system', 'Sorry, I encountered an error. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle voice recording
  const handleMicrophoneClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
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
        await handleVoiceMessage(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setIsVoiceMode(true);
      setUiMinimized(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      addMessage('system', 'Could not access microphone. Please check your permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Handle voice message (simulate transcription)
  const handleVoiceMessage = async (audioBlob) => {
    setIsProcessing(true);
    
    // Simulate voice transcription
    const simulatedTranscription = "Hello, this is a simulated voice message transcription.";
    
    try {
      // Add transcribed user message
      addMessage('user', simulatedTranscription, { isVoice: true });
      
      // Process the message
      await handleTextMessage(simulatedTranscription);
      
    } catch (error) {
      console.error('Error processing voice message:', error);
      addMessage('system', 'Error processing voice input. Please try again.');
    } finally {
      setIsProcessing(false);
      setIsVoiceMode(false);
      setUiMinimized(false);
    }
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      handleTextMessage(inputText);
      setInputText('');
    }
  };

  // Clear conversation
  const clearConversation = () => {
    setMessages([]);
  };

  // Export conversation
  const exportConversation = () => {
    const conversationData = {
      messages: messages,
      timestamp: new Date().toISOString(),
      voiceConfig: voiceConfig
    };
    
    const blob = new Blob([JSON.stringify(conversationData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `moegpt-conversation-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="app">
      {/* Audio element for playing responses */}
      <audio ref={audioRef} preload="none" />
      
      {/* Header - always visible */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <h1 className="logo">MoeGPT</h1>
            <p className="subtitle">Your AI Voice Assistant</p>
          </div>
          
          <div className="status-section">
            <div className="connection-status">
              <div className={`status-dot ${connectionStatus}`}></div>
              <span className={`status-text ${connectionStatus}`}>
                {connectionStatus}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main">
        {!uiMinimized ? (
          // Full UI Mode
          <div className="chat-container">
            {/* Left Panel - Controls */}
            <div className="left-panel">
              <div className="controls-sections">
                {/* Voice Configuration */}
                <div className="control-section">
                  <h3 className="section-title">Voice Settings</h3>
                  <div className="voice-settings">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={voiceConfig.voiceEnabled}
                        onChange={(e) => setVoiceConfig(prev => ({ ...prev, voiceEnabled: e.target.checked }))}
                      />
                      <span>Enable Voice Responses</span>
                    </label>
                    
                    <div className="voice-type-section">
                      <label className="select-label">Voice Type</label>
                      <select
                        value={voiceConfig.voiceType}
                        onChange={(e) => setVoiceConfig(prev => ({ ...prev, voiceType: e.target.value }))}
                        className="voice-select"
                      >
                        <option value="female">Female Voice</option>
                        <option value="male">Male Voice</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Microphone Control */}
                <div className="control-section">
                  <h3 className="section-title">Voice Input</h3>
                  <button
                    onClick={handleMicrophoneClick}
                    disabled={isProcessing}
                    className={`mic-button ${isRecording ? 'recording' : ''} ${isProcessing ? 'disabled' : ''}`}
                  >
                    {isRecording ? '🛑' : '🎤'}
                  </button>
                  {isRecording && (
                    <p className="recording-text">
                      🎤 Listening... Click to stop
                    </p>
                  )}
                </div>

                {/* Conversation Controls */}
                <div className="control-section">
                  <h3 className="section-title">Controls</h3>
                  <div className="control-buttons">
                    <button
                      onClick={clearConversation}
                      className="clear-button"
                    >
                      Clear Chat
                    </button>
                    <button
                      onClick={exportConversation}
                      disabled={messages.length === 0}
                      className="export-button"
                    >
                      Export Chat
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Panel - Chat Interface */}
            <div className="right-panel">
              {/* Messages Area */}
              <div className="messages-area">
                {messages.length === 0 && (
                  <div className="welcome-section">
                    <h2 className="welcome-title">Welcome to MoeGPT! 👋</h2>
                    <p className="welcome-text">I'm your AI voice assistant. You can:</p>
                    <ul className="welcome-list">
                      <li>• Type messages or use voice input</li>
                      <li>• Ask me to open websites (YouTube, Google, etc.)</li>
                      <li>• Search for anything online</li>
                      <li>• Get the current time</li>
                      <li>• Check the weather</li>
                    </ul>
                    <p className="welcome-example">Try saying: "Open YouTube" or "Search for cats"</p>
                  </div>
                )}
                
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message-container ${message.role}`}
                  >
                    <div className={`message ${message.role} ${message.role === 'system' ? 'system' : ''}`}>
                      <div className="message-header">
                        <span className="message-author">
                          {message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'MoeGPT'}
                          {message.isVoice && ' 🎤'}
                        </span>
                        <span className="message-time">
                          {formatTime(message.timestamp)}
                        </span>
                      </div>
                      <p className="message-content">{message.content}</p>
                      {message.actionExecuted && (
                        <div className="action-indicator">
                          Action: {message.actionExecuted}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {isProcessing && (
                  <div className="message-container assistant">
                    <div className="message assistant">
                      <div className="processing-indicator">
                        <div className="spinner"></div>
                        <span>MoeGPT is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Chat Input */}
              <div className="chat-input-container">
                <form onSubmit={handleSubmit} className="chat-form">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    disabled={isProcessing || isRecording}
                    placeholder={isRecording ? "Listening..." : "Type your message..."}
                    className="chat-input"
                  />
                  <button
                    type="submit"
                    disabled={isProcessing || isRecording || !inputText.trim()}
                    className="send-button"
                  >
                    Send
                  </button>
                </form>
              </div>
            </div>
          </div>
        ) : (
          // Minimized Voice Mode UI
          <div className="voice-mode">
            {/* Logo */}
            <div className="voice-mode-logo">
              <h1 className="voice-mode-title">MoeGPT</h1>
              <p className="voice-mode-subtitle">Listening...</p>
            </div>

            {/* Voice Visualizer */}
            <div className="voice-visualizer">
              <button
                onClick={handleMicrophoneClick}
                className={`voice-mode-mic ${isRecording ? 'recording' : ''}`}
              >
                {isRecording ? '🛑' : '🎤'}
              </button>
            </div>

            {/* Latest Messages */}
            <div className="voice-mode-messages">
              <div className="messages-panel">
                <div className="recent-messages">
                  {messages.slice(-3).map((message) => (
                    <div
                      key={message.id}
                      className={`voice-message-container ${message.role}`}
                    >
                      <div className={`voice-message ${message.role}`}>
                        <div className="voice-message-author">
                          {message.role === 'user' ? 'You' : 'MoeGPT'}
                          {message.isVoice && ' 🎤'}
                        </div>
                        <p className="voice-message-content">{message.content}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Exit Voice Mode */}
            <button
              onClick={() => {
                setUiMinimized(false);
                setIsVoiceMode(false);
                if (isRecording) {
                  stopRecording();
                }
              }}
              className="exit-voice-mode"
            >
              Exit Voice Mode
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;