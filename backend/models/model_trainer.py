import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
import logging
from typing import List, Dict, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoeGPTTrainer:
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the MoeGPT trainer
        
        Recommended base models:
        - microsoft/DialoGPT-small: Fastest, lower quality
        - microsoft/DialoGPT-medium: Good balance (default)
        - microsoft/DialoGPT-large: Best quality, slower
        - distilgpt2: Lightweight alternative
        - gpt2: Classic option
        """
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Add custom tokens for MoeGPT
        special_tokens = {
            "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def load_training_data(self, jsonl_file: str) -> List[Dict]:
        """Load training data from JSONL file"""
        training_data = []
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Validate required fields
                        if 'instruction' not in data or 'response' not in data:
                            logger.warning(f"Line {line_num}: Missing required fields")
                            continue
                            
                        training_data.append(data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue
                        
            logger.info(f"Loaded {len(training_data)} training examples")
            return training_data
            
        except FileNotFoundError:
            logger.error(f"Training data file not found: {jsonl_file}")
            return []
    
    def create_conversation_format(self, data: List[Dict]) -> List[str]:
        """Convert training data to conversation format"""
        conversations = []
        
        for item in data:
            # Create a conversation format similar to DialoGPT
            conversation = f"<|user|>{item['instruction']}<|assistant|>{item['response']}<|endoftext|>"
            conversations.append(conversation)
            
        return conversations
    
    def tokenize_data(self, conversations: List[str]) -> Dataset:
        """Tokenize conversations for training"""
        logger.info("Tokenizing training data...")
        
        # Tokenize all conversations
        tokenized = self.tokenizer(
            conversations,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone()  # For language modeling
        })
        
        return dataset
    
    def train(
        self,
        jsonl_file: str,
        output_dir: str = "./models/moegpt_model",
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        use_early_stopping: bool = True,
        patience: int = 3
    ):
        """
        Train MoeGPT on custom data
        
        Args:
            jsonl_file: Path to training data
            output_dir: Where to save the trained model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            warmup_steps: Warmup steps for learning rate scheduler
            save_steps: Save model every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay for optimizer
            use_early_stopping: Whether to use early stopping
            patience: Patience for early stopping
        """
        
        # Load and prepare data
        logger.info("Loading training data...")
        training_data = self.load_training_data(jsonl_file)
        
        if not training_data:
            logger.error("No training data loaded!")
            return
        
        # Convert to conversation format
        conversations = self.create_conversation_format(training_data)
        
        # Split data (80% train, 20% eval)
        split_idx = int(0.8 * len(conversations))
        train_conversations = conversations[:split_idx]
        eval_conversations = conversations[split_idx:]
        
        logger.info(f"Train samples: {len(train_conversations)}")
        logger.info(f"Eval samples: {len(eval_conversations)}")
        
        # Tokenize data
        train_dataset = self.tokenize_data(train_conversations)
        eval_dataset = self.tokenize_data(eval_conversations) if eval_conversations else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=use_early_stopping and eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            save_total_limit=3,  # Keep only 3 checkpoints
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )
        
        # Callbacks
        callbacks = []
        if use_early_stopping and eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        
        # Start training
        logger.info("Starting training...")
        try:
            trainer.train()
            
            # Save final model
            logger.info(f"Saving final model to {output_dir}")
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training info
            training_info = {
                "base_model": self.base_model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_samples": len(train_conversations),
                "eval_samples": len(eval_conversations) if eval_conversations else 0,
            }
            
            with open(os.path.join(output_dir, "training_info.json"), "w") as f:
                json.dump(training_info, f, indent=2)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def create_sample_data(self, output_file: str = "./data/training_data.jsonl"):
        """Create sample training data if none exists"""
        sample_data = [
            {"instruction": "Hello", "response": "Hi there! I'm MoeGPT, your AI assistant. How can I help you today?"},
            {"instruction": "How are you?", "response": "I'm doing well, thank you for asking! I'm here and ready to help you with anything you need."},
            {"instruction": "What's your name?", "response": "I'm MoeGPT, your personal AI voice assistant! I'm designed to help you with various tasks and have conversations."},
            {"instruction": "Open YouTube", "response": "I'll open YouTube for you right now!"},
            {"instruction": "What time is it?", "response": "Let me check the current time for you."},
            {"instruction": "Tell me a joke", "response": "Here's a joke for you: Why don't scientists trust atoms? Because they make up everything!"},
            {"instruction": "Search for Python tutorials", "response": "I'll search for Python tutorials on Google for you!"},
            {"instruction": "Play music", "response": "I'll start your music player for you. Enjoy your tunes!"},
            {"instruction": "Good morning", "response": "Good morning! I hope you're having a wonderful day. What can I help you with?"},
            {"instruction": "Thank you", "response": "You're very welcome! I'm always happy to help. Is there anything else you'd like assistance with?"},
            {"instruction": "Goodbye", "response": "Goodbye! It was great talking with you. Feel free to come back anytime you need help!"},
            {"instruction": "What can you do?", "response": "I can help you with many things! I can open websites, search the web, tell you the time, play music, and have conversations with you. Just ask me anything!"},
        ]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write sample data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created sample training data at {output_file}")

def main():
    """Command line interface for training"""
    parser = argparse.ArgumentParser(description="Train MoeGPT on custom data")
    parser.add_argument("--data", default="./data/training_data.jsonl", help="Path to training data JSONL file")
    parser.add_argument("--output", default="./models/moegpt_model", help="Output directory for trained model")
    parser.add_argument("--base-model", default="microsoft/DialoGPT-medium", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample training data")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        trainer = MoeGPTTrainer(args.base_model)
        trainer.create_sample_data(args.data)
        return
    
    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"Training data file not found: {args.data}")
        logger.info("Use --create-sample-data to create sample training data")
        return
    
    # Initialize trainer and start training
    trainer = MoeGPTTrainer(args.base_model)
    trainer.train(
        jsonl_file=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()