import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoeGPTModel:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", custom_model_path: str = None):
        """
        Initialize MoeGPT with a local model
        
        Recommended models:
        - microsoft/DialoGPT-medium (conversational)
        - distilgpt2 (lightweight)
        - microsoft/DialoGPT-small (fastest)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if custom_model_path and os.path.exists(custom_model_path):
            logger.info(f"Loading custom model from {custom_model_path}")
            self.model_name = custom_model_path
        else:
            logger.info(f"Loading base model: {model_name}")
            self.model_name = model_name
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        
    def generate_response(self, user_input: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate a response to user input"""
        try:
            # Format input for conversation
            input_text = f"User: {user_input}\nMoeGPT:"
            
            # Tokenize input
            inputs = self.tokenizer.encode_plus(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (MoeGPT's response)
            if "MoeGPT:" in full_response:
                response = full_response.split("MoeGPT:")[-1].strip()
                # Clean up any remaining "User:" parts
                if "User:" in response:
                    response = response.split("User:")[0].strip()
            else:
                response = full_response[len(input_text):].strip()
                
            return response if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."
    
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
    # Initialize model
    moegpt = MoeGPTModel("microsoft/DialoGPT-medium")
    
    # Train on custom data (optional)
    # moegpt.train_model("../data/training_data.jsonl")
    
    # Test the model
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = moegpt.generate_response(user_input)
        print(f"MoeGPT: {response}")