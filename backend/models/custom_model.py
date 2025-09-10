import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
import re
from typing import List, Dict, Optional
import logging
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedMoeGPTModel:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", custom_model_path: str = None, use_openai: bool = False):
        """
        Initialize MoeGPT with improved handling and OpenAI integration
        
        Args:
            model_name: Base model name (e.g., "microsoft/DialoGPT-medium")
            custom_model_path: Path to custom trained model
            use_openai: Whether to use OpenAI API instead of local model
        """
        self.use_openai = use_openai
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
        
        if use_openai:
            self._init_openai()
        else:
            self._init_local_model(model_name, custom_model_path)
    
    def _init_openai(self):
        """Initialize OpenAI API client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key or api_key == 'your_openai_key_here':
                logger.error("OpenAI API key not found or not set properly in .env file")
                self.openai_client = None
                return
            
            openai.api_key = api_key
            self.openai_client = openai
            
            # Test the connection
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info("OpenAI API connection successful")
            except Exception as e:
                logger.warning(f"OpenAI API test failed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.openai_client = None
    
    def _init_local_model(self, model_name: str, custom_model_path: str):
        """Initialize local transformer model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            if custom_model_path and os.path.exists(custom_model_path):
                logger.info(f"Loading custom model from {custom_model_path}")
                self.model_name = custom_model_path
            else:
                logger.info(f"Loading base model: {model_name}")
                self.model_name = model_name
                
            # Load tokenizer and model with error handling
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.to(self.device)
            logger.info("Local model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _clean_input_text(self, text: str) -> str:
        """Clean and prepare input text for processing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove or replace problematic characters
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Ensure text ends with punctuation for better model response
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _prepare_conversation_context(self, user_input: str) -> str:
        """Prepare conversation context with history"""
        # Clean input
        clean_input = self._clean_input_text(user_input)
        
        if not self.conversation_history:
            # First conversation
            context = f"User: {clean_input}\nMoeGPT:"
        else:
            # Build context with recent history
            context_parts = []
            
            # Add recent history (limit to prevent token overflow)
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                context_parts.append(f"User: {exchange['user']}")
                context_parts.append(f"MoeGPT: {exchange['assistant']}")
            
            # Add current input
            context_parts.append(f"User: {clean_input}")
            context_parts.append("MoeGPT:")
            
            context = "\n".join(context_parts)
        
        return context
    
    def _generate_openai_response(self, user_input: str) -> str:
        """Generate response using OpenAI API"""
        if not self.openai_client:
            return "I'm sorry, but I'm having trouble connecting to my AI service right now."
        
        try:
            # Prepare conversation history for OpenAI
            messages = [
                {
                    "role": "system", 
                    "content": "You are MoeGPT, a helpful AI voice assistant. Respond naturally and conversationally. Keep responses concise but helpful."
                }
            ]
            
            # Add conversation history
            for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
            
            # Add current input
            messages.append({"role": "user", "content": user_input})
            
            logger.info(f"Sending to OpenAI: {user_input}")
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            assistant_response = response.choices[0].message.content.strip()
            logger.info(f"OpenAI response: {assistant_response}")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Could you try again?"
    
    def _generate_local_response(self, user_input: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate response using local model"""
        if not self.model or not self.tokenizer:
            return "I'm sorry, my AI model isn't available right now."
        
        try:
            # Prepare conversation context
            context = self._prepare_conversation_context(user_input)
            logger.info(f"Model input context: {context}")
            
            # Tokenize input with proper handling
            inputs = self.tokenizer.encode_plus(
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
                add_special_tokens=True
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate response with improved parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_length,
                    min_length=input_ids.shape[1] + 5,  # Ensure minimum response length
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model raw output: {full_response}")
            
            # Extract only the new part (MoeGPT's response)
            response = self._extract_assistant_response(full_response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating local response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def _extract_assistant_response(self, full_response: str, context: str) -> str:
        """Extract the assistant's response from the full generated text"""
        try:
            # Find the last "MoeGPT:" in the response
            if "MoeGPT:" in full_response:
                # Split by "MoeGPT:" and get the last part
                parts = full_response.split("MoeGPT:")
                response = parts[-1].strip()
                
                # Remove any "User:" parts that might have been generated
                if "User:" in response:
                    response = response.split("User:")[0].strip()
                
                # Remove any remaining context
                response = response.replace(context, "").strip()
                
            else:
                # Fallback: extract everything after the input context
                response = full_response.replace(context, "").strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Ensure we have a valid response
            if not response or len(response.strip()) < 3:
                return "I'm not sure how to respond to that. Could you rephrase your question?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            return "I'm having trouble formulating a response right now."
    
    def _clean_response(self, response: str) -> str:
        """Clean and improve the assistant's response"""
        if not response:
            return ""
        
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response.strip())
        
        # Remove common artifacts from generation
        response = re.sub(r'^(MoeGPT:|User:|Assistant:)', '', response).strip()
        response = re.sub(r'<\|.*?\|>', '', response)  # Remove special tokens
        
        # Remove repetitive patterns
        words = response.split()
        if len(words) > 4:
            # Check for immediate repetition
            for i in range(len(words) - 1):
                if i + 2 < len(words) and words[i] == words[i + 2]:
                    # Found repetition, truncate
                    response = ' '.join(words[:i + 1])
                    break
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure response ends with punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response
    
    def generate_response(self, user_input: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate a response to user input with improved handling"""
        try:
            # Clean and validate input
            clean_input = self._clean_input_text(user_input)
            if not clean_input:
                return "I didn't catch that. Could you please repeat your question?"
            
            logger.info(f"Processing user input: '{clean_input}'")
            
            # Generate response based on configured method
            if self.use_openai:
                response = self._generate_openai_response(clean_input)
            else:
                response = self._generate_local_response(clean_input, max_length, temperature)
            
            # Update conversation history
            self._update_conversation_history(clean_input, response)
            
            logger.info(f"Generated response: '{response}'")
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return "I'm sorry, I'm having trouble right now. Could you try asking me something else?"
    
    def _update_conversation_history(self, user_input: str, assistant_response: str):
        """Update conversation history"""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        logger.info("Clearing conversation history")
        self.conversation_history = []
    
    def set_openai_mode(self, use_openai: bool):
        """Switch between OpenAI and local model"""
        logger.info(f"Switching to {'OpenAI' if use_openai else 'local'} model")
        self.use_openai = use_openai
        if use_openai and not hasattr(self, 'openai_client'):
            self._init_openai()
    
    def get_model_info(self) -> Dict:
        """Get information about the current model setup"""
        return {
            "using_openai": self.use_openai,
            "openai_available": hasattr(self, 'openai_client') and self.openai_client is not None,
            "local_model_available": hasattr(self, 'model') and self.model is not None,
            "conversation_history_length": len(self.conversation_history),
            "device": str(self.device) if hasattr(self, 'device') else "N/A",
            "model_name": getattr(self, 'model_name', 'Unknown')
        }
    
    # Training methods (keeping from original)
    def prepare_training_data(self, jsonl_file: str) -> Dataset:
        """Prepare training data from JSONL file"""
        conversations = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                conversation = f"User: {data['instruction']}\nMoeGPT: {data['response']}<|endoftext|>"
                conversations.append(conversation)
        
        # Tokenize conversations
        tokenized_data = self.tokenizer(
            conversations,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
            "labels": tokenized_data["input_ids"].clone()
        })
        
        return dataset
    
    def train_model(self, jsonl_file: str, output_dir: str = "./models/moegpt_model", epochs: int = 3):
        """Fine-tune the model on custom data"""
        if self.use_openai:
            logger.warning("Cannot train OpenAI model. Switch to local model first.")
            return
        
        if not self.model or not self.tokenizer:
            logger.error("Local model not available for training")
            return
        
        logger.info("Preparing training data...")
        train_dataset = self.prepare_training_data(jsonl_file)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
            report_to=None,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed!")
    
    def load_custom_model(self, model_path: str):
        """Load a previously trained custom model"""
        if self.use_openai:
            logger.warning("Cannot load custom model while in OpenAI mode")
            return
        
        logger.info(f"Loading custom model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.to(self.device)
        logger.info("Custom model loaded successfully!")

# Usage example:
if __name__ == "__main__":
    # Test with local model
    print("Testing local model...")
    moegpt_local = ImprovedMoeGPTModel("microsoft/DialoGPT-medium", use_openai=False)
    
    # Test with OpenAI (if API key is available)
    print("Testing OpenAI model...")
    moegpt_openai = ImprovedMoeGPTModel(use_openai=True)
    
    # Interactive test
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        # Try OpenAI first, fallback to local
        if moegpt_openai.openai_client:
            response = moegpt_openai.generate_response(user_input)
        else:
            response = moegpt_local.generate_response(user_input)
        
        print(f"MoeGPT: {response}")