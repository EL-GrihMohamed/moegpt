from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import tempfile
import logging
import asyncio
from typing import Optional, Dict, Any, List

# Import our improved modules
from models.custom_model import ImprovedMoeGPTModel
from api.voice_handler import ImprovedVoiceHandler
from utils.actions import ActionHandler

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MoeGPT API",
    description="Local AI Voice Assistant API with Enhanced Voice Processing",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
moegpt_model = None
voice_handler = None
action_handler = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    voice_enabled: bool = False
    voice_type: str = "female"
    use_openai: bool = False

class ChatResponse(BaseModel):
    response: str
    action_executed: bool = False
    action_result: Optional[str] = None
    audio_file: Optional[str] = None
    transcription_confidence: Optional[float] = None

class VoiceChatResponse(BaseModel):
    transcribed_text: str
    response_text: str
    action_executed: bool = False
    audio_file: Optional[str] = None
    processing_time: Optional[float] = None
    errors: Optional[List[str]] = None

class ModelConfigRequest(BaseModel):
    use_openai: bool = False
    model_name: str = "microsoft/DialoGPT-medium"
    clear_history: bool = False

class VoiceConfigRequest(BaseModel):
    voice_type: str = "female"
    use_whisper: bool = True

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and handlers on startup"""
    global moegpt_model, voice_handler, action_handler
    
    logger.info("🚀 Starting MoeGPT API v2.0...")
    
    try:
        # Initialize AI model with fallback logic
        logger.info("📚 Loading AI model...")
        custom_model_path = "./models/moegpt_model"
        
        # Try OpenAI first if API key is available
        use_openai = os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_key_here'
        
        if use_openai:
            logger.info("🌐 Initializing with OpenAI support...")
            moegpt_model = ImprovedMoeGPTModel(use_openai=True)
            # Also initialize local model as fallback
            if os.path.exists(custom_model_path):
                moegpt_model.set_openai_mode(False)
                moegpt_model = ImprovedMoeGPTModel(custom_model_path=custom_model_path, use_openai=False)
                moegpt_model.set_openai_mode(True)
            else:
                moegpt_model._init_local_model("microsoft/DialoGPT-medium", None)
        else:
            logger.info("🖥️ Initializing with local model...")
            if os.path.exists(custom_model_path):
                moegpt_model = ImprovedMoeGPTModel(custom_model_path=custom_model_path, use_openai=False)
            else:
                moegpt_model = ImprovedMoeGPTModel("microsoft/DialoGPT-medium", use_openai=False)
        
        # Initialize voice handler
        logger.info("🎤 Initializing enhanced voice handler...")
        voice_handler = ImprovedVoiceHandler(use_whisper=True, voice_type="female")
        
        # Initialize action handler
        logger.info("⚡ Initializing action handler...")
        action_handler = ActionHandler()
        
        # Log model info
        model_info = moegpt_model.get_model_info()
        logger.info(f"🤖 Model Info: {model_info}")
        
        logger.info("✅ MoeGPT API started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize MoeGPT: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global voice_handler
    logger.info("🛑 Shutting down MoeGPT API...")
    if voice_handler:
        voice_handler.cleanup()

# Enhanced health check endpoint
@app.get("/")
async def root():
    """Enhanced health check endpoint"""
    model_info = moegpt_model.get_model_info() if moegpt_model else {}
    
    return {
        "message": "MoeGPT API v2.0 is running!",
        "status": "healthy",
        "components": {
            "model_loaded": moegpt_model is not None,
            "voice_enabled": voice_handler is not None,
            "actions_enabled": action_handler is not None,
        },
        "model_info": model_info,
        "endpoints": {
            "chat": "/chat",
            "voice_chat": "/voice_chat", 
            "transcribe": "/transcribe",
            "train": "/train",
            "configure": "/configure"
        }
    }

# Enhanced chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with better error handling"""
    if not moegpt_model:
        raise HTTPException(status_code=500, detail="AI model not loaded")
    
    processing_start = asyncio.get_event_loop().time()
    
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"💬 Chat request: '{user_message[:100]}...'")
        
        # Switch model mode if requested
        if request.use_openai != moegpt_model.use_openai:
            moegpt_model.set_openai_mode(request.use_openai)
        
        # Check if this is an action command
        action_executed = False
        action_result = None
        
        if action_handler:
            action_executed, action_result = await action_handler.execute_action(user_message)
            logger.info(f"⚡ Action executed: {action_executed}, Result: {action_result}")
        
        # Generate AI response
        if action_executed and action_result != "I didn't understand that command.":
            ai_response = action_result
        else:
            ai_response = moegpt_model.generate_response(user_message)
        
        # Generate audio if voice is enabled
        audio_file = None
        if request.voice_enabled and voice_handler:
            voice_handler.set_voice_type(request.voice_type)
            
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_result = await voice_handler.text_to_speech(ai_response, temp_audio.name)
            if audio_result:
                audio_file = temp_audio.name
                logger.info(f"🔊 Audio generated: {audio_file}")
        
        processing_time = asyncio.get_event_loop().time() - processing_start
        logger.info(f"⏱️ Chat processed in {processing_time:.2f}s")
        
        return ChatResponse(
            response=ai_response,
            action_executed=action_executed,
            action_result=action_result if action_executed else None,
            audio_file=audio_file
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Enhanced voice transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Enhanced audio transcription with better error handling"""
    if not voice_handler:
        raise HTTPException(status_code=500, detail="Voice handler not available")
    
    logger.info(f"🎤 Transcription request: {audio_file.filename}")
    
    try:
        # Validate file
        if audio_file.size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        if audio_file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")
        
        # Save uploaded audio to temporary file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        content = await audio_file.read()
        with open(temp_audio.name, "wb") as f:
            f.write(content)
        
        logger.info(f"📁 Audio saved temporarily: {temp_audio.name}")
        
        # Transcribe audio
        transcribed_text = await voice_handler.speech_to_text(temp_audio.name)
        
        # Clean up
        try:
            os.unlink(temp_audio.name)
        except:
            pass
        
        if not transcribed_text:
            return JSONResponse(
                status_code=400, 
                content={
                    "error": "Could not transcribe audio",
                    "suggestions": [
                        "Ensure audio is clear and audible",
                        "Check microphone settings", 
                        "Try speaking more slowly",
                        "Reduce background noise"
                    ]
                }
            )
        
        logger.info(f"✅ Transcription successful: '{transcribed_text}'")
        return {"text": transcribed_text}
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# ENHANCED VOICE CHAT ENDPOINT - The main fix!
@app.post("/voice_chat", response_model=VoiceChatResponse)
async def enhanced_voice_chat(
    audio_file: UploadFile = File(...),
    voice_type: str = Form("female"),
    enable_actions: bool = Form(True),
    use_openai: bool = Form(False)
):
    """
    🎯 ENHANCED Complete voice chat pipeline with comprehensive error handling
    
    This is the main endpoint that should fix your voice chat issues!
    """
    if not voice_handler or not moegpt_model:
        raise HTTPException(status_code=500, detail="Voice handler or model not available")
    
    processing_start = asyncio.get_event_loop().time()
    errors = []
    
    logger.info("🗣️ ==> ENHANCED VOICE CHAT PIPELINE STARTED <==")
    
    try:
        # === STEP 1: VALIDATE AND SAVE AUDIO ===
        logger.info("📥 Step 1: Processing uploaded audio...")
        
        if not audio_file or audio_file.size == 0:
            raise HTTPException(status_code=400, detail="No audio file provided or file is empty")
        
        if audio_file.size > 15 * 1024 * 1024:  # 15MB limit
            raise HTTPException(status_code=400, detail="Audio file too large")
        
        # Save uploaded audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            content = await audio_file.read()
            with open(temp_audio.name, "wb") as f:
                f.write(content)
            logger.info(f"✅ Audio saved: {temp_audio.name} ({len(content)} bytes)")
        except Exception as e:
            errors.append(f"Failed to save audio: {e}")
            raise HTTPException(status_code=500, detail="Failed to save audio file")
        
        # === STEP 2: TRANSCRIBE SPEECH TO TEXT ===
        logger.info("🔍 Step 2: Transcribing speech to text...")
        
        try:
            user_text = await voice_handler.speech_to_text(temp_audio.name)
            logger.info(f"🎯 RAW TRANSCRIPTION: '{user_text}'")
        except Exception as e:
            errors.append(f"Transcription error: {e}")
            user_text = None
        finally:
            # Clean up audio file
            try:
                os.unlink(temp_audio.name)
            except:
                pass
        
        # Validate transcription
        if not user_text or len(user_text.strip()) < 2:
            fallback_response = "Sorry, I didn't catch that. Can you repeat what you said?"
            logger.warning(f"⚠️ Transcription failed or too short: '{user_text}'")
            
            # Generate fallback audio
            temp_response_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            voice_handler.set_voice_type(voice_type)
            await voice_handler.text_to_speech(fallback_response, temp_response_audio.name)
            
            return VoiceChatResponse(
                transcribed_text=user_text or "",
                response_text=fallback_response,
                action_executed=False,
                audio_file=temp_response_audio.name,
                processing_time=asyncio.get_event_loop().time() - processing_start,
                errors=errors
            )
        
        logger.info(f"✅ CLEAN TRANSCRIPTION: '{user_text}'")
        
        # === STEP 3: SWITCH MODEL MODE IF NEEDED ===
        if use_openai != moegpt_model.use_openai:
            logger.info(f"🔄 Switching model mode to: {'OpenAI' if use_openai else 'Local'}")
            moegpt_model.set_openai_mode(use_openai)
        
        # === STEP 4: CHECK FOR SYSTEM ACTIONS ===
        logger.info("⚡ Step 4: Checking for system actions...")
        
        action_executed = False
        action_result = None
        
        if enable_actions and action_handler:
            try:
                action_executed, action_result = await action_handler.execute_action(user_text)
                logger.info(f"🎯 ACTION RESULT: executed={action_executed}, result='{action_result}'")
            except Exception as e:
                errors.append(f"Action processing error: {e}")
                logger.error(f"❌ Action error: {e}")
        
        # === STEP 5: GENERATE AI RESPONSE ===
        logger.info("🤖 Step 5: Generating AI response...")
        
        try:
            if action_executed and action_result and action_result != "I didn't understand that command.":
                ai_response = action_result
                logger.info("✅ Using action result as response")
            else:
                ai_response = moegpt_model.generate_response(user_text)
                logger.info(f"🎯 AI RESPONSE: '{ai_response}'")
                
                # Fallback if AI response is poor
                if not ai_response or len(ai_response.strip()) < 5:
                    ai_response = "I'm not sure how to respond to that. Could you ask me something else?"
                    errors.append("AI generated empty or very short response")
                
        except Exception as e:
            ai_response = "I'm having trouble processing your request right now. Could you try again?"
            errors.append(f"AI generation error: {e}")
            logger.error(f"❌ AI generation error: {e}")
        
        # === STEP 6: GENERATE SPEECH RESPONSE ===
        logger.info("🔊 Step 6: Converting response to speech...")
        
        temp_response_audio = None
        try:
            temp_response_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            voice_handler.set_voice_type(voice_type)
            
            audio_result = await voice_handler.text_to_speech(ai_response, temp_response_audio.name)
            
            if not audio_result:
                errors.append("Failed to generate speech audio")
                logger.warning("⚠️ TTS failed")
            else:
                logger.info(f"✅ Audio response generated: {temp_response_audio.name}")
                
        except Exception as e:
            errors.append(f"TTS error: {e}")
            logger.error(f"❌ TTS error: {e}")
        
        # === STEP 7: RETURN RESULTS ===
        processing_time = asyncio.get_event_loop().time() - processing_start
        logger.info(f"⏱️ TOTAL PROCESSING TIME: {processing_time:.2f}s")
        logger.info("🎉 ==> VOICE CHAT PIPELINE COMPLETED <==")
        
        return VoiceChatResponse(
            transcribed_text=user_text,
            response_text=ai_response,
            action_executed=action_executed,
            audio_file=temp_response_audio.name if temp_response_audio and audio_result else None,
            processing_time=processing_time,
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Critical voice chat error: {e}")
        processing_time = asyncio.get_event_loop().time() - processing_start
        
        # Generate error response audio
        error_response = "I'm sorry, I encountered an error. Please try again."
        temp_error_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        try:
            await voice_handler.text_to_speech(error_response, temp_error_audio.name)
        except:
            temp_error_audio = None
        
        return VoiceChatResponse(
            transcribed_text="",
            response_text=error_response,
            action_executed=False,
            audio_file=temp_error_audio.name if temp_error_audio else None,
            processing_time=processing_time,
            errors=[f"Critical error: {str(e)}"]
        )

# Model configuration endpoint
@app.post("/configure_model")
async def configure_model(config: ModelConfigRequest):
    """Configure model settings"""
    global moegpt_model
    
    try:
        if config.clear_history and moegpt_model:
            moegpt_model.clear_conversation_history()
            logger.info("🧹 Conversation history cleared")
        
        if moegpt_model:
            moegpt_model.set_openai_mode(config.use_openai)
            logger.info(f"🔄 Model mode set to: {'OpenAI' if config.use_openai else 'Local'}")
        
        return {
            "message": "Model configured successfully",
            "model_info": moegpt_model.get_model_info() if moegpt_model else {}
        }
        
    except Exception as e:
        logger.error(f"❌ Model configuration error: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

# Voice configuration endpoint  
@app.post("/configure_voice")
async def configure_voice(config: VoiceConfigRequest):
    """Configure voice settings"""
    global voice_handler
    
    try:
        if voice_handler:
            voice_handler.cleanup()
        
        voice_handler = ImprovedVoiceHandler(
            use_whisper=config.use_whisper,
            voice_type=config.voice_type
        )
        
        logger.info(f"🎤 Voice configured: {config.voice_type}, Whisper: {config.use_whisper}")
        
        return {"message": f"Voice configured: {config.voice_type}, Whisper: {config.use_whisper}"}
        
    except Exception as e:
        logger.error(f"❌ Voice configuration error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice configuration failed: {str(e)}")

# Audio file serving endpoint
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        filename,
        media_type="audio/wav",
        filename=os.path.basename(filename)
    )

# Enhanced model status endpoint
@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    model_info = moegpt_model.get_model_info() if moegpt_model else {}
    
    return {
        "system_status": "operational",
        "components": {
            "ai_model": {
                "loaded": moegpt_model is not None,
                "info": model_info
            },
            "voice_handler": {
                "loaded": voice_handler is not None,
                "whisper_available": hasattr(voice_handler, 'whisper_model') and voice_handler.whisper_model is not None if voice_handler else False,
                "tts_available": hasattr(voice_handler, 'tts_engine') and voice_handler.tts_engine is not None if voice_handler else False
            },
            "action_handler": {
                "loaded": action_handler is not None,
                "actions_count": len(action_handler.actions) if action_handler else 0
            }
        },
        "files": {
            "custom_model_exists": os.path.exists("./models/moegpt_model"),
            "training_data_exists": os.path.exists("./data/training_data.jsonl"),
            "env_file_exists": os.path.exists(".env")
        },
        "api_keys": {
            "openai_configured": bool(os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_key_here'),
            "elevenlabs_configured": bool(os.getenv('ELEVENLABS_API_KEY') and os.getenv('ELEVENLABS_API_KEY') != 'your_elevenlabs_key_here')
        }
    }

# Training endpoint (enhanced)
@app.post("/train")
async def train_model(
    epochs: int = Form(3),
    model_name: str = Form("microsoft/DialoGPT-medium"),
    use_existing_data: bool = Form(True)
):
    """Enhanced model training endpoint"""
    global moegpt_model
    
    if not moegpt_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if moegpt_model.use_openai:
        raise HTTPException(status_code=400, detail="Cannot train OpenAI model. Switch to local model first.")
    
    training_data_path = "./data/training_data.jsonl"
    if not os.path.exists(training_data_path):
        raise HTTPException(status_code=400, detail="Training data file not found")
    
    try:
        logger.info(f"🏋️ Starting training with {epochs} epochs...")
        
        # Train the model
        moegpt_model.train_model(
            jsonl_file=training_data_path,
            epochs=epochs
        )
        
        # Reload the trained model
        custom_model_path = "./models/moegpt_model"
        moegpt_model.load_custom_model(custom_model_path)
        
        logger.info("✅ Model training completed successfully!")
        return {"message": "Model training completed successfully!", "epochs": epochs}
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Available actions endpoint
@app.get("/available_actions")
async def available_actions():
    """Get list of available actions with examples"""
    if not action_handler:
        raise HTTPException(status_code=500, detail="Action handler not available")
    
    return {
        "actions": list(action_handler.actions.keys()),
        "examples": [
            "open YouTube",
            "search for Python tutorials", 
            "what time is it?",
            "open calculator",
            "visit github.com",
            "play music",
            "tell me the date",
            "open notepad",
            "lock screen"
        ],
        "categories": {
            "web": ["open_youtube", "open_google", "search_google", "open_website"],
            "system": ["tell_time", "tell_date", "open_calculator", "open_notepad", "open_file_manager"],
            "media": ["play_music", "pause_music", "volume_up", "volume_down"],
            "control": ["lock_screen", "shutdown", "restart"]
        }
    }

# Debug endpoint for testing voice pipeline components
@app.post("/debug/test_voice_pipeline")
async def test_voice_pipeline():
    """Test voice pipeline components individually"""
    results = {
        "tts_test": False,
        "whisper_test": False,
        "model_test": False,
        "action_test": False,
        "errors": []
    }
    
    # Test TTS
    try:
        if voice_handler and voice_handler.tts_engine:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            result = await voice_handler.text_to_speech("This is a test", temp_audio.name)
            results["tts_test"] = bool(result)
            if result:
                try:
                    os.unlink(temp_audio.name)
                except:
                    pass
        else:
            results["errors"].append("TTS engine not available")
    except Exception as e:
        results["errors"].append(f"TTS test failed: {e}")
    
    # Test Whisper/STT availability
    try:
        if voice_handler:
            results["whisper_test"] = hasattr(voice_handler, 'whisper_model') and voice_handler.whisper_model is not None
        else:
            results["errors"].append("Voice handler not available")
    except Exception as e:
        results["errors"].append(f"Whisper test failed: {e}")
    
    # Test AI model
    try:
        if moegpt_model:
            test_response = moegpt_model.generate_response("Hello")
            results["model_test"] = bool(test_response and len(test_response) > 0)
        else:
            results["errors"].append("AI model not available")
    except Exception as e:
        results["errors"].append(f"Model test failed: {e}")
    
    # Test actions
    try:
        if action_handler:
            success, result = await action_handler.execute_action("what time is it")
            results["action_test"] = success
        else:
            results["errors"].append("Action handler not available")
    except Exception as e:
        results["errors"].append(f"Action test failed: {e}")
    
    return results

# Conversation history endpoint
@app.get("/conversation_history")
async def get_conversation_history():
    """Get current conversation history"""
    if not moegpt_model:
        raise HTTPException(status_code=500, detail="Model not available")
    
    return {
        "history": moegpt_model.conversation_history,
        "length": len(moegpt_model.conversation_history),
        "max_length": moegpt_model.max_history
    }

@app.delete("/conversation_history")
async def clear_conversation_history():
    """Clear conversation history"""
    if not moegpt_model:
        raise HTTPException(status_code=500, detail="Model not available")
    
    moegpt_model.clear_conversation_history()
    logger.info("🧹 Conversation history cleared via API")
    
    return {"message": "Conversation history cleared"}

# Requirements and setup info endpoint
@app.get("/setup_info")
async def get_setup_info():
    """Get setup information and requirements"""
    return {
        "required_packages": [
            "fastapi", "uvicorn", "transformers", "torch", "whisper", 
            "pyttsx3", "soundfile", "librosa", "pydub", "openai", "python-dotenv"
        ],
        "optional_packages": [
            "vosk", "pyaudio", "speech_recognition"
        ],
        "setup_steps": [
            "1. Install required packages: pip install -r requirements.txt",
            "2. Set up .env file with API keys",
            "3. Download Whisper model (automatic on first use)",
            "4. Optional: Download Vosk model for offline STT",
            "5. Run: uvicorn main:app --reload"
        ],
        "file_structure": {
            ".env": "API keys and configuration",
            "data/training_data.jsonl": "Training data for custom model",
            "models/moegpt_model/": "Custom trained model (created after training)",
            "api/voice_handler.py": "Voice processing logic",
            "models/custom_model.py": "AI model logic",
            "utils/actions.py": "System action handlers"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )