import os
import tempfile
import logging
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# TTS (Text-to-Speech)
import pyttsx3

# STT (Speech-to-Text) - using OpenAI Whisper (local)
import whisper
import soundfile as sf
import numpy as np

# Alternative STT using SpeechRecognition + vosk
try:
    import speech_recognition as sr
    from vosk import Model, KaldiRecognizer
    import pyaudio
    import json
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceHandler:
    def __init__(self, use_whisper: bool = True, voice_type: str = "female"):
        """
        Initialize voice handler with local TTS and STT
        
        Args:
            use_whisper: Use Whisper for STT (True) or Vosk (False)
            voice_type: "male" or "female"
        """
        self.use_whisper = use_whisper
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize TTS
        self._init_tts(voice_type)
        
        # Initialize STT
        if use_whisper:
            self._init_whisper()
        else:
            self._init_vosk()
    
    def _init_tts(self, voice_type: str):
        """Initialize pyttsx3 for text-to-speech"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            
            # Set voice based on preference
            selected_voice = None
            for voice in voices:
                if voice_type.lower() == "female" and "female" in voice.name.lower():
                    selected_voice = voice.id
                    break
                elif voice_type.lower() == "male" and "male" in voice.name.lower():
                    selected_voice = voice.id
                    break
            
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            logger.info("TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_engine = None
    
    def _init_whisper(self):
        """Initialize Whisper for speech-to-text"""
        try:
            # Load Whisper model (base is good balance of speed/accuracy)
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper STT initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.whisper_model = None
    
    def _init_vosk(self):
        """Initialize Vosk for speech-to-text (alternative to Whisper)"""
        if not VOSK_AVAILABLE:
            logger.error("Vosk not available. Install with: pip install vosk pyaudio")
            return
            
        try:
            # Download model if not exists
            model_path = "vosk-model-en-us-0.22"
            if not os.path.exists(model_path):
                logger.warning(f"Vosk model not found at {model_path}")
                logger.info("Download from: https://alphacephei.com/vosk/models")
                return
                
            self.vosk_model = Model(model_path)
            self.vosk_rec = KaldiRecognizer(self.vosk_model, 16000)
            logger.info("Vosk STT initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            self.vosk_model = None
    
    async def text_to_speech(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Convert text to speech"""
        if not self.tts_engine:
            logger.error("TTS engine not available")
            return None
            
        try:
            def _speak():
                if output_file:
                    self.tts_engine.save_to_file(text, output_file)
                    self.tts_engine.runAndWait()
                    return output_file
                else:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    return "spoken"
            
            # Run TTS in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _speak)
            return result
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    async def speech_to_text_whisper(self, audio_file: str) -> Optional[str]:
        """Convert speech to text using Whisper"""
        if not self.whisper_model:
            logger.error("Whisper model not available")
            return None
            
        try:
            def _transcribe():
                result = self.whisper_model.transcribe(audio_file)
                return result["text"].strip()
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(self.executor, _transcribe)
            
            logger.info(f"Transcribed: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            return None
    
    async def speech_to_text_vosk(self, audio_data: bytes) -> Optional[str]:
        """Convert speech to text using Vosk"""
        if not hasattr(self, 'vosk_rec'):
            logger.error("Vosk not available")
            return None
            
        try:
            def _transcribe():
                if self.vosk_rec.AcceptWaveform(audio_data):
                    result = json.loads(self.vosk_rec.Result())
                    return result.get('text', '')
                else:
                    partial = json.loads(self.vosk_rec.PartialResult())
                    return partial.get('partial', '')
            
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(self.executor, _transcribe)
            
            logger.info(f"Transcribed: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Vosk STT error: {e}")
            return None
    
    async def speech_to_text(self, audio_file: str) -> Optional[str]:
        """Main STT method - uses configured STT engine"""
        if self.use_whisper:
            return await self.speech_to_text_whisper(audio_file)
        else:
            # For Vosk, we need to convert file to appropriate format
            try:
                audio_data, sample_rate = sf.read(audio_file)
                if sample_rate != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                
                # Convert to bytes
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                return await self.speech_to_text_vosk(audio_bytes)
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                return None
    
    def set_voice_type(self, voice_type: str):
        """Change voice type (male/female)"""
        self._init_tts(voice_type)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'tts_engine') and self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        self.executor.shutdown(wait=False)

# Test/Example usage
async def test_voice_handler():
    """Test the voice handler"""
    vh = VoiceHandler(use_whisper=True, voice_type="female")
    
    # Test TTS
    print("Testing TTS...")
    await vh.text_to_speech("Hello! I am MoeGPT, your AI voice assistant.")
    
    # Test STT (you would need to provide an actual audio file)
    # print("Testing STT...")
    # text = await vh.speech_to_text("test_audio.wav")
    # print(f"Recognized: {text}")
    
    vh.cleanup()

if __name__ == "__main__":
    asyncio.run(test_voice_handler())