import os
import tempfile
import logging
from typing import Optional, List
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

# Enhanced Voice Handler class with better error handling and features
class ImprovedVoiceHandler(VoiceHandler):
    """Enhanced version of VoiceHandler with additional features"""
    
    def __init__(self, use_whisper: bool = True, voice_type: str = "female", model_size: str = "base"):
        """
        Initialize improved voice handler
        
        Args:
            use_whisper: Use Whisper for STT
            voice_type: "male" or "female"
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self.confidence_threshold = 0.5
        super().__init__(use_whisper, voice_type)
    
    def _init_whisper(self):
        """Initialize Whisper with configurable model size"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info(f"Whisper {self.model_size} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper {self.model_size}: {e}")
            # Fallback to base model
            try:
                logger.info("Falling back to base model...")
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper base model loaded as fallback")
            except Exception as fallback_e:
                logger.error(f"Fallback to base model also failed: {fallback_e}")
                self.whisper_model = None
    
    async def speech_to_text_whisper(self, audio_file: str) -> Optional[str]:
        """Enhanced Whisper transcription with confidence scoring"""
        if not self.whisper_model:
            logger.error("Whisper model not available")
            return None
            
        try:
            def _transcribe():
                # Transcribe with additional options
                result = self.whisper_model.transcribe(
                    audio_file,
                    language="en",
                    task="transcribe",
                    fp16=False,
                    verbose=False
                )
                
                text = result["text"].strip()
                
                # Calculate average confidence from segments
                if "segments" in result and result["segments"]:
                    confidences = []
                    for segment in result["segments"]:
                        if "avg_logprob" in segment:
                            # Convert log probability to confidence (approximate)
                            confidence = min(1.0, max(0.0, (segment["avg_logprob"] + 1.0)))
                            confidences.append(confidence)
                    
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        logger.info(f"Transcription confidence: {avg_confidence:.2f}")
                        
                        # Return None if confidence is too low
                        if avg_confidence < self.confidence_threshold:
                            logger.warning(f"Low confidence transcription rejected: {avg_confidence:.2f}")
                            return None
                
                return text
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(self.executor, _transcribe)
            
            if text:
                logger.info(f"Transcribed: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Enhanced Whisper STT error: {e}")
            return None
    
    async def text_to_speech(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Enhanced TTS with better error handling"""
        if not self.tts_engine:
            logger.error("TTS engine not available")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return None
            
        try:
            # Clean text for better speech
            cleaned_text = self._clean_text_for_speech(text)
            
            def _speak():
                if output_file:
                    self.tts_engine.save_to_file(cleaned_text, output_file)
                    self.tts_engine.runAndWait()
                    
                    # Verify file was created
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                        return output_file
                    else:
                        logger.error("TTS file creation failed or file is empty")
                        return None
                else:
                    self.tts_engine.say(cleaned_text)
                    self.tts_engine.runAndWait()
                    return "spoken"
            
            # Run TTS in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _speak)
            return result
            
        except Exception as e:
            logger.error(f"Enhanced TTS error: {e}")
            return None
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common abbreviations
        replacements = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
            r'\bvs\.': 'versus',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove or replace problematic characters
        text = text.replace('&', ' and ')
        text = re.sub(r'[^\w\s\.,!?;:\-\'"()]', ' ', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for transcription"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Transcription confidence threshold set to: {self.confidence_threshold}")
    
    def get_available_voices(self) -> List[dict]:
        """Get list of available TTS voices"""
        if not self.tts_engine:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            voice_list = []
            
            for i, voice in enumerate(voices):
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages if hasattr(voice, 'languages') else [],
                    'gender': voice.gender if hasattr(voice, 'gender') else 'unknown',
                    'age': voice.age if hasattr(voice, 'age') else 'unknown'
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def set_speech_properties(self, rate: int = 180, volume: float = 0.9):
        """Set TTS speech properties"""
        if not self.tts_engine:
            logger.error("TTS engine not available")
            return False
        
        try:
            self.tts_engine.setProperty('rate', max(50, min(300, rate)))
            self.tts_engine.setProperty('volume', max(0.0, min(1.0, volume)))
            logger.info(f"Speech properties set: rate={rate}, volume={volume}")
            return True
        except Exception as e:
            logger.error(f"Error setting speech properties: {e}")
            return False
    
    async def test_audio_pipeline(self) -> dict:
        """Test the complete audio pipeline"""
        results = {
            'tts_available': False,
            'stt_available': False,
            'whisper_model': None,
            'available_voices': 0,
            'errors': []
        }
        
        # Test TTS
        try:
            if self.tts_engine:
                results['tts_available'] = True
                results['available_voices'] = len(self.get_available_voices())
            else:
                results['errors'].append("TTS engine not initialized")
        except Exception as e:
            results['errors'].append(f"TTS test error: {e}")
        
        # Test STT
        try:
            if self.whisper_model:
                results['stt_available'] = True
                results['whisper_model'] = self.model_size
            else:
                results['errors'].append("Whisper model not loaded")
        except Exception as e:
            results['errors'].append(f"STT test error: {e}")
        
        return results

# Test/Example usage
async def test_voice_handler():
    """Test the enhanced voice handler"""
    vh = ImprovedVoiceHandler(use_whisper=True, voice_type="female", model_size="base")
    
    # Test pipeline
    test_results = await vh.test_audio_pipeline()
    print("Pipeline Test Results:", test_results)
    
    # Test TTS
    print("Testing enhanced TTS...")
    await vh.text_to_speech("Hello! I am MoeGPT, your enhanced AI voice assistant.")
    
    # Test available voices
    voices = vh.get_available_voices()
    print(f"Available voices: {len(voices)}")
    for voice in voices[:3]:  # Show first 3
        print(f"  - {voice['name']} ({voice['gender']})")
    
    vh.cleanup()

if __name__ == "__main__":
    asyncio.run(test_voice_handler())