import React, { useState, useEffect, useRef } from 'react';
import ChatInterface from './components/ChatInterface';
import VoiceSelector from './components/VoiceSelector';
import MicrophoneButton from './components/MicrophoneButton';
import ConversationLog from './components/ConversationLog';
import { apiService } from './services/api';
import { voiceService } from './services/voiceService';
import './styles/globals.css';

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
  const [modelStatus, setModelStatus] = useState({});
  const [isTraining, setIsTraining] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  // Refs
  const audioRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Check backend connection on mount
  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const checkBackendStatus = async () => {
    try {
      const status = await apiService.getModelStatus();
      setModelStatus(status);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Backend connection failed:', error);
      setConnectionStatus('disconnected');
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

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

  const handleTextMessage = async (text) => {
    if (!text.trim()) return;

    // Add user message
    addMessage('user', text);
    setIsProcessing(true);

    try {
      const response = await apiService.sendMessage({
        message: text,
        voice_enabled: voiceConfig.voiceEnabled,
        voice_type: voiceConfig.voiceType
      });

      // Add AI response
      addMessage('assistant', response.response, {
        actionExecuted: response.action_executed,
        actionResult: response.action_result,
        audioFile: response.audio_file
      });

      // Play audio if available
      if (response.audio_file && voiceConfig.voiceEnabled) {
        await playAudioResponse(response.audio_file);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('system', 'Sorry, I encountered an error. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVoiceMessage = async (audioBlob) => {
    setIsProcessing(true);
    addMessage('system', 'Processing voice message...', { isTemp: true });

    try {
      // Convert blob to file
      const audioFile = new File([audioBlob], 'voice_input.wav', { type: 'audio/wav' });
      
      // Use complete voice chat pipeline
      const response = await apiService.voiceChat(audioFile, {
        voice_type: voiceConfig.voiceType,
        enable_actions: true
      });

      // Remove temp message
      setMessages(prev => prev.filter(msg => !msg.isTemp));

      // Add transcribed user message
      if (response.transcribed_text) {
        addMessage('user', response.transcribed_text, { isVoice: true });
      }

      // Add AI response
      addMessage('assistant', response.response_text, {
        actionExecuted: response.action_executed,
        isVoice: true,
        audioFile: response.audio_file
      });

      // Play audio response
      if (response.audio_file) {
        await playAudioResponse(response.audio_file);
      }

    } catch (error) {
          console.error('Error processing voice message:', error);
        } finally {
          setIsProcessing(false);
        }
      };
    }