from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import tempfile
import logging
import asyncio
from typing import Optional, Dict, Any

# Import our custom modules
from models.custom_model import MoeGPTModel
from api.voice_handler import VoiceHandler
from utils.actions import ActionHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MoeGPT API",
    description="Local AI Voice Assistant API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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

class ChatResponse(BaseModel):
    response: str
    action_executed: bool = False
    action_result: Optional[str] = None
    audio_file: Optional[str] = None

class TrainRequest(BaseModel):
    epochs: int = 3
    model_name: str = "microsoft/DialoGPT-medium"

class VoiceConfigRequest(BaseModel):
    voice_type: str = "female"
    use_whisper: bool = True

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and handlers on startup"""
    global moegpt_model, voice_handler, action_handler
    
    logger.info("Starting MoeGPT API...")
    
    try:
        # Initialize AI model
        logger.info("Loading AI model...")
        custom_model_path = "./models/moegpt_model"
        if os.path.exists(custom_model_path):
            moegpt_model = MoeGPTModel(custom_model_path=custom_model_path)
        else:
            moegpt_model = MoeGPTModel("microsoft/DialoGPT-medium")
        
        # Initialize voice handler
        logger.info("Initializing voice handler...")
        voice_handler = VoiceHandler(use_whisper=True, voice_type="female")
        
        # Initialize action handler
        logger.info("Initializing action handler...")
        action_handler = ActionHandler()
        
        logger.info("MoeGPT API started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize MoeGPT: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global voice_handler
    if voice_handler:
        voice_handler.cleanup()

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MoeGPT API is running!",
        "status": "healthy",
        "model_loaded": moegpt_model is not None,
        "voice_enabled": voice_handler is not None,
        "actions_enabled": action_handler is not None
    }

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not moegpt_model:
        raise HTTPException(status_code=500, detail="AI model not loaded")
    
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Check if this is an action command
        action_executed = False
        action_result = None
        
        if action_handler:
            action_executed, action_result = await action_handler.execute_action(user_message)
        
        # Generate AI response
        if action_executed:
            # Use action result as the response
            ai_response = action_result
        else:
            # Generate normal AI response
            ai_response = moegpt_model.generate_response(user_message)
        
        # Generate audio if voice is enabled
        audio_file = None
        if request.voice_enabled and voice_handler:
            # Set voice type if different
            if voice_handler.tts_engine:
                voice_handler.set_voice_type(request.voice_type)
            
            # Generate audio file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_result = await voice_handler.text_to_speech(ai_response, temp_audio.name)
            if audio_result:
                audio_file = temp_audio.name
        
        return ChatResponse(
            response=ai_response,
            action_executed=action_executed,
            action_result=action_result if action_executed else None,
            audio_file=audio_file
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Voice transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio to text"""
    if not voice_handler:
        raise HTTPException(status_code=500, detail="Voice handler not available")
    
    try:
        # Save uploaded audio to temporary file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        with open(temp_audio.name, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Transcribe audio
        transcribed_text = await voice_handler.speech_to_text(temp_audio.name)
        
        # Clean up
        os.unlink(temp_audio.name)
        
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        return {"text": transcribed_text}
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# Voice configuration endpoint
@app.post("/configure_voice")
async def configure_voice(config: VoiceConfigRequest):
    """Configure voice settings"""
    global voice_handler
    
    try:
        if voice_handler:
            voice_handler.cleanup()
        
        voice_handler = VoiceHandler(
            use_whisper=config.use_whisper,
            voice_type=config.voice_type
        )
        
        return {"message": f"Voice configured: {config.voice_type}, Whisper: {config.use_whisper}"}
        
    except Exception as e:
        logger.error(f"Voice configuration error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice configuration failed: {str(e)}")

# Audio file serving endpoint
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    file_path = filename
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=os.path.basename(file_path)
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

# Training endpoint
@app.post("/train")
async def train_model(request: TrainRequest):
    """Train the model on custom data"""
    global moegpt_model
    
    training_data_path = "./data/training_data.jsonl"
    if not os.path.exists(training_data_path):
        raise HTTPException(status_code=400, detail="Training data file not found")
    
    try:
        # Initialize a fresh model for training
        logger.info(f"Starting training with {request.epochs} epochs...")
        
        if moegpt_model:
            # Train the existing model
            moegpt_model.train_model(
                jsonl_file=training_data_path,
                epochs=request.epochs
            )
        else:
            # Create new model and train
            moegpt_model = MoeGPTModel(request.model_name)
            moegpt_model.train_model(
                jsonl_file=training_data_path,
                epochs=request.epochs
            )
        
        # Reload the trained model
        custom_model_path = "./models/moegpt_model"
        moegpt_model.load_custom_model(custom_model_path)
        
        return {"message": "Model training completed successfully!"}
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Model status endpoint
@app.get("/model_status")
async def model_status():
    """Get current model status"""
    return {
        "model_loaded": moegpt_model is not None,
        "voice_handler_ready": voice_handler is not None,
        "action_handler_ready": action_handler is not None,
        "custom_model_exists": os.path.exists("./models/moegpt_model"),
        "training_data_exists": os.path.exists("./data/training_data.jsonl")
    }

# Available actions endpoint
@app.get("/available_actions")
async def available_actions():
    """Get list of available actions"""
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
            "tell me the date"
        ]
    }

# Conversation endpoint for real-time voice chat
@app.post("/voice_chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    voice_type: str = Form("female"),
    enable_actions: bool = Form(True)
):
    """Complete voice chat pipeline: STT -> AI -> Actions -> TTS"""
    if not voice_handler or not moegpt_model:
        raise HTTPException(status_code=500, detail="Voice handler or model not available")
    
    try:
        # Save uploaded audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_audio.name, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Transcribe speech to text
        user_text = await voice_handler.speech_to_text(temp_audio.name)
        os.unlink(temp_audio.name)
        
        if not user_text:
            raise HTTPException(status_code=400, detail="Could not understand speech")
        
        # Process with AI and actions
        action_executed = False
        action_result = None
        
        if enable_actions and action_handler:
            action_executed, action_result = await action_handler.execute_action(user_text)
        
        if action_executed:
            ai_response = action_result
        else:
            ai_response = moegpt_model.generate_response(user_text)
        
        # Generate speech response
        temp_response_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        voice_handler.set_voice_type(voice_type)
        audio_result = await voice_handler.text_to_speech(ai_response, temp_response_audio.name)
        
        return {
            "transcribed_text": user_text,
            "response_text": ai_response,
            "action_executed": action_executed,
            "audio_file": temp_response_audio.name if audio_result else None
        }
        
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )