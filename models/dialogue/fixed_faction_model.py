"""
Fixed Faction Dialogue Model
Simplified architecture that preserves text generation quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, GPT2LMHeadModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np

# Import path fix
import sys
sys.path.append(os.getcwd())

from config.config import DIALOGUE_CONFIG, FACTIONS, DEVICE_CONFIG

class FixedDialogueModelConfig:
    """Simplified configuration that preserves generation quality"""
    def __init__(self):
        # Use GPT2 instead of DialoGPT for better generation
        self.base_model = "gpt2"  # More reliable generation
        self.max_sequence_length = 512
        self.vocab_size = 50257
        
        # Simplified conditioning - less intrusive
        self.faction_embedding_dim = 32  # Smaller, less interference
        self.num_factions = 4
        self.relationship_embedding_dim = 16  # Smaller
        self.num_relationship_states = 6
        
        # Training parameters
        self.learning_rate = 5e-5  # Slower, more stable
        self.warmup_steps = 200
        self.max_epochs = 2  # Fewer epochs to prevent overfitting
        self.batch_size = 2  # Smaller batches for stability
        self.gradient_accumulation_steps = 8  # Effective batch size = 16
        
        # Generation parameters
        self.use_mixed_precision = True
        self.max_new_tokens = 100
        self.temperature = 0.7  # More conservative
        self.top_p = 0.9
        self.repetition_penalty = 1.05  # Lighter penalty

class SimpleFactionDataset(Dataset):
    """Simplified dataset with better formatting"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self._load_and_process_data(data_path)
        
        self.faction_to_id = {
            "ashvattha": 0, "vaikuntha": 1, 
            "yuga_striders": 2, "shroud_mantra": 3
        }
        
        self.relationship_to_id = {
            "hostile": 0, "suspicious": 1, "neutral": 2,
            "respectful": 3, "trusting": 4, "allied": 5
        }
    
    def _load_and_process_data(self, data_path: str):
        """Load and process conversation data with better formatting"""
        
        faction_files = [
            "ashvattha_conversations.json", "vaikuntha_conversations.json", 
            "yuga_striders_conversations.json", "shroud_mantra_conversations.json"
        ]
        
        for faction_file in faction_files:
            file_path = os.path.join(data_path, faction_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for conversation in data['conversations']:
                    self._process_conversation(conversation)
                    
                print(f"Processed {len(data['conversations'])} conversations from {faction_file}")
    
    def _process_conversation(self, conversation: Dict):
        """Process conversation with cleaner formatting"""
        
        dialogue = conversation['dialogue']
        faction = conversation['npc_faction']
        relationship = conversation['relationship_context']
        
        # Create training examples from consecutive player-NPC pairs
        for i in range(0, len(dialogue) - 1, 2):
            if (i < len(dialogue) - 1 and 
                dialogue[i]['speaker'] == 'player' and 
                dialogue[i + 1]['speaker'] == 'npc'):
                
                player_text = dialogue[i]['text']
                npc_text = dialogue[i + 1]['text']
                
                # Create clean input-output format
                # Format: [FACTION:ashvattha] [REL:neutral] Player: What is karma? Assistant: 
                input_text = f"[FACTION:{faction}] [REL:{relationship}] Player: {player_text} Assistant:"
                target_text = f" {npc_text}"
                
                # Combine for language modeling
                full_text = input_text + target_text + self.tokenizer.eos_token
                
                self.examples.append({
                    'text': full_text,
                    'input_text': input_text,
                    'target_text': target_text,
                    'faction': faction,
                    'relationship': relationship,
                    'input_length': len(input_text)
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the full text
        encoding = self.tokenizer(
            example['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for language modeling)
        labels = encoding['input_ids'].clone()
        
        # Mask the input portion for loss calculation (only train on NPC response)
        input_tokens = self.tokenizer(example['input_text'])['input_ids']
        input_length = len(input_tokens)
        labels[0, :input_length] = -100  # Ignore input tokens in loss
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'faction_id': torch.tensor(self.faction_to_id[example['faction']], dtype=torch.long),
            'relationship_id': torch.tensor(self.relationship_to_id[example['relationship']], dtype=torch.long),
        }

class SimpleFactionModel(nn.Module):
    """Simplified model that doesn't interfere with generation quality"""
    
    def __init__(self, config: FixedDialogueModelConfig):
        super().__init__()
        self.config = config
        
        # Use standard GPT2 model
        self.base_model = GPT2LMHeadModel.from_pretrained(config.base_model)
        
        # Resize embeddings to include our special tokens if needed
        # (We're using text formatting instead of embedding conditioning)
        
    def forward(self, input_ids, attention_mask, labels=None, faction_id=None, relationship_id=None, **kwargs):
        """Simple forward pass using base GPT2"""
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate_response(self, input_ids, attention_mask, max_new_tokens=100, 
                         temperature=0.7, top_p=0.9, repetition_penalty=1.05):
        """Generate response using GPT2's built-in generation"""
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.base_model.config.eos_token_id,
                eos_token_id=self.base_model.config.eos_token_id
            )
        
        return outputs

class FixedDialogueTrainer:
    """Simplified trainer for the fixed model"""
    
    def __init__(self, model: SimpleFactionModel, config: FixedDialogueModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Setup optimizer for base model parameters only
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if config.use_mixed_precision else None
    
    def train(self, train_dataset, save_path: str = "models/dialogue/fixed_faction_model.pt"):
        """Train the simplified model"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        total_steps = len(train_loader) * self.config.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Training simplified model on {self.device}")
        print(f"Total steps: {total_steps}")
        
        self.model.train()
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{self.config.max_epochs}, "
                          f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__
        }, save_path)
        
        print(f"Fixed model saved to {save_path}")

class FixedDialogueInference:
    """Inference system for the fixed model"""
    
    def __init__(self, model_path: str = "models/dialogue/fixed_faction_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        config = FixedDialogueModelConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = SimpleFactionModel(config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Fixed model loaded from {model_path}")
    
    def generate_faction_response(self, player_input: str, faction: str, relationship: str) -> str:
        """Generate NPC response with faction conditioning"""
        
        # Format input using the same format as training
        input_text = f"[FACTION:{faction}] [REL:{relationship}] Player: {player_input} Assistant:"
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate_response(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05
            )
        
        # Decode full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the NPC response part
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response[len(input_text):].strip()
        
        return response

def train_fixed_model():
    """Train the fixed faction dialogue model"""
    
    print("=== Training Fixed Faction Dialogue Model ===")
    
    config = FixedDialogueModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = SimpleFactionDataset("data/synthetic/dialogue", tokenizer)
    print(f"Loaded {len(dataset)} training examples")
    
    # Create model and trainer
    model = SimpleFactionModel(config)
    trainer = FixedDialogueTrainer(model, config)
    
    # Train
    trainer.train(dataset)
    
    print("Fixed model training complete!")

def test_fixed_model():
    """Test the fixed model"""
    
    print("=== Testing Fixed Model ===")
    
    try:
        inference = FixedDialogueInference()
        
        # Test scenarios
        tests = [
            ("What is karma?", "ashvattha", "neutral"),
            ("Is reincarnation real?", "vaikuntha", "hostile"),
            ("How do we break free from suffering?", "yuga_striders", "trusting"),
            ("What if reality is just a story?", "shroud_mantra", "respectful")
        ]
        
        for player_input, faction, relationship in tests:
            response = inference.generate_faction_response(player_input, faction, relationship)
            print(f"\n🎭 {faction} ({relationship})")
            print(f"💬 Player: {player_input}")
            print(f"🤖 NPC: {response}")
        
    except Exception as e:
        print(f"Testing failed: {e}")

if __name__ == "__main__":
    # Uncomment to retrain with fixed architecture
    train_fixed_model()
    
    # Test the fixed model
    test_fixed_model()