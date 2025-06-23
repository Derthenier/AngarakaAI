"""
Faction Dialogue Model Architecture for Threads of Kaliyuga
Transformer-based model for generating faction-specific philosophical responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np
from enum import Enum

# Import our framework
# Import our framework modules
import sys
import os
sys.path.append(os.getcwd())

from config.config import DIALOGUE_CONFIG, FACTIONS, DEVICE_CONFIG

class DialogueModelConfig:
    """Configuration for the faction dialogue model"""
    def __init__(self):
        # Model architecture
        self.base_model = "microsoft/DialoGPT-small"  # Good for dialogue, small enough for fast inference
        self.max_sequence_length = 512
        self.vocab_size = 50257  # DialoGPT vocab size
        
        # Faction-specific embeddings
        self.faction_embedding_dim = 64
        self.num_factions = 4
        
        # Relationship awareness
        self.relationship_embedding_dim = 32
        self.num_relationship_states = 6  # hostile, suspicious, neutral, respectful, trusting, allied
        
        # Training parameters
        self.learning_rate = 2e-5
        self.warmup_steps = 500
        self.max_epochs = 3
        self.batch_size = 8  # Small batch for your hardware
        self.gradient_accumulation_steps = 2  # Effective batch size = 16
        
        # Inference optimization
        self.use_mixed_precision = True
        self.max_new_tokens = 150
        self.temperature = 0.8
        self.top_p = 0.9
        self.repetition_penalty = 1.1

@dataclass
class DialogueInput:
    """Input data structure for dialogue model"""
    conversation_history: List[str]  # Previous messages in conversation
    current_player_input: str        # What player just said
    npc_faction: str                 # Which faction the NPC belongs to
    relationship_status: str         # Current relationship level
    topic_context: str               # Main philosophical topic being discussed
    conversation_tone: str           # Expected tone (friendly, hostile, etc.)

class FactionDialogueDataset(Dataset):
    """Dataset for training faction dialogue models"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        
        # Load and process conversation data
        self._load_conversations(data_path)
        
        # Create faction and relationship mappings
        self.faction_to_id = {
            "ashvattha": 0,
            "vaikuntha": 1, 
            "yuga_striders": 2,
            "shroud_mantra": 3
        }
        
        self.relationship_to_id = {
            "hostile": 0,
            "suspicious": 1,
            "neutral": 2,
            "respectful": 3,
            "trusting": 4,
            "allied": 5
        }
    
    def _load_conversations(self, data_path: str):
        """Load conversations from JSON files"""
        faction_files = [
            "ashvattha_conversations.json",
            "vaikuntha_conversations.json", 
            "yuga_striders_conversations.json",
            "shroud_mantra_conversations.json"
        ]
        
        for faction_file in faction_files:
            file_path = os.path.join(data_path, faction_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for conversation in data['conversations']:
                    self._process_conversation(conversation)
                    
                print(f"Loaded {len(data['conversations'])} conversations from {faction_file}")
    
    def _process_conversation(self, conversation: Dict):
        """Process a single conversation into training examples"""
        dialogue = conversation['dialogue']
        
        # Extract metadata
        faction = conversation['npc_faction']
        relationship = conversation['relationship_context']
        topic = conversation['main_topic']
        
        # Create training examples from dialogue pairs
        conversation_history = []
        
        for i, exchange in enumerate(dialogue):
            if exchange['speaker'] == 'npc' and i > 0:
                # Get previous player input
                player_input = dialogue[i-1]['text']
                npc_response = exchange['text']
                
                # Create training example
                example = {
                    'conversation_history': conversation_history.copy(),
                    'player_input': player_input,
                    'npc_response': npc_response,
                    'faction': faction,
                    'relationship': relationship,
                    'topic': topic,
                    'emotion': exchange['emotion'],
                    'faction_hints': exchange.get('faction_alignment_hints', [])
                }
                
                self.conversations.append(example)
            
            # Update conversation history
            conversation_history.append(f"{exchange['speaker']}: {exchange['text']}")
            
            # Keep history manageable
            if len(conversation_history) > 6:  # Last 3 exchanges
                conversation_history = conversation_history[-6:]
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """Get a training example"""
        example = self.conversations[idx]
        
        # Format input text
        history_text = " ".join(example['conversation_history'][-4:])  # Last 2 exchanges
        input_text = f"Context: {history_text} Player: {example['player_input']}"
        target_text = example['npc_response']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length - 150,  # Leave room for response
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=150,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        # Combine for language modeling
        full_text = input_text + " NPC: " + target_text
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': full_encoding['input_ids'].squeeze(),  # For language modeling
            'faction_id': torch.tensor(self.faction_to_id[example['faction']], dtype=torch.long),
            'relationship_id': torch.tensor(self.relationship_to_id[example['relationship']], dtype=torch.long),
            'input_length': len(input_encoding['input_ids'][0]),  # Where NPC response starts
        }

class FactionDialogueModel(nn.Module):
    """Transformer model with faction and relationship conditioning"""
    
    def __init__(self, config: DialogueModelConfig):
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.transformer_config = AutoConfig.from_pretrained(config.base_model)
        self.transformer = AutoModel.from_pretrained(config.base_model)
        
        # Faction conditioning
        self.faction_embedding = nn.Embedding(
            config.num_factions, 
            config.faction_embedding_dim
        )
        
        # Relationship conditioning  
        self.relationship_embedding = nn.Embedding(
            config.num_relationship_states,
            config.relationship_embedding_dim
        )
        
        # Projection layers
        self.condition_projection = nn.Linear(
            config.faction_embedding_dim + config.relationship_embedding_dim,
            self.transformer_config.hidden_size
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(
            self.transformer_config.hidden_size,
            config.vocab_size,
            bias=False
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize custom layer weights"""
        nn.init.normal_(self.faction_embedding.weight, std=0.02)
        nn.init.normal_(self.relationship_embedding.weight, std=0.02)
        nn.init.normal_(self.condition_projection.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask, faction_id, relationship_id, labels=None, input_length=None, **kwargs):
        """Forward pass with faction and relationship conditioning"""
        
        # Get base transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        
        # Get conditioning embeddings
        faction_emb = self.faction_embedding(faction_id)  # [batch_size, faction_dim]
        relationship_emb = self.relationship_embedding(relationship_id)  # [batch_size, rel_dim]
        
        # Combine conditioning
        condition_emb = torch.cat([faction_emb, relationship_emb], dim=-1)  # [batch_size, total_dim]
        condition_proj = self.condition_projection(condition_emb)  # [batch_size, hidden_size]
        
        # Add conditioning to hidden states
        condition_proj = condition_proj.unsqueeze(1)  # [batch_size, 1, hidden_size]
        conditioned_hidden = hidden_states + condition_proj  # Broadcast across sequence
        
        # Language modeling head
        logits = self.lm_head(conditioned_hidden)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            # Calculate loss only on the NPC response part
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs['loss'] = loss
        
        return outputs
    
    def generate_response(self, input_ids, attention_mask, faction_id, relationship_id, 
                         max_new_tokens=150, temperature=0.8, top_p=0.9, 
                         repetition_penalty=1.1):
        """Generate NPC response with conditioning"""
        
        self.eval()
        with torch.no_grad():
            # Start with input sequence
            generated = input_ids.clone()
            past_key_values = None
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    faction_id=faction_id,
                    relationship_id=relationship_id
                )
                
                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(generated.shape[0]):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i][indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ], dim=-1)
                
                # Check for EOS token (you might want to define this)
                if next_token.item() == 50256:  # EOS token for GPT-2
                    break
            
            return generated

class DialogueTrainer:
    """Training pipeline for faction dialogue models"""
    
    def __init__(self, model: FactionDialogueModel, config: DialogueModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup mixed precision if available
        self.scaler = torch.amp.GradScaler('cuda') if config.use_mixed_precision else None
    
    def train(self, train_dataset: FactionDialogueDataset, 
              val_dataset: FactionDialogueDataset = None,
              save_path: str = "models/dialogue/faction_dialogue_model.pt"):
        """Train the dialogue model"""
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues on Windows
        )
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.config.max_epochs
        
        # Setup learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Starting training on {self.device}")
        print(f"Total training steps: {total_steps}")
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                        
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Logging
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config.max_epochs}, "
                          f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if save_path:
                checkpoint_path = save_path.replace('.pt', f'_epoch_{epoch+1}.pt')
                self.save_model(checkpoint_path)
        
        print("Training completed!")
        if save_path:
            self.save_model(save_path)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class DialogueInference:
    """Optimized inference pipeline for real-time dialogue"""
    
    def __init__(self, model_path: str, tokenizer_name: str = "microsoft/DialoGPT-small"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        config = DialogueModelConfig()
        self.model = FactionDialogueModel(config)
        self.load_model(model_path)
        self.model.eval()
        
        # Faction and relationship mappings
        self.faction_to_id = {
            "ashvattha": 0, "vaikuntha": 1, 
            "yuga_striders": 2, "shroud_mantra": 3
        }
        self.relationship_to_id = {
            "hostile": 0, "suspicious": 1, "neutral": 2,
            "respectful": 3, "trusting": 4, "allied": 5
        }
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Inference model loaded from {path}")
    
    def generate_npc_response(self, dialogue_input: DialogueInput) -> str:
        """Generate NPC response for gameplay"""
        
        # Format input
        history_text = " ".join(dialogue_input.conversation_history[-4:])
        input_text = f"Context: {history_text} Player: {dialogue_input.current_player_input}"
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=350,  # Leave room for response
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get faction and relationship IDs
        faction_id = torch.tensor([self.faction_to_id[dialogue_input.npc_faction]], device=self.device)
        relationship_id = torch.tensor([self.relationship_to_id[dialogue_input.relationship_status]], device=self.device)
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate_response(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                faction_id=faction_id,
                relationship_id=relationship_id,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9
            )
        
        # Decode response (only the new tokens)
        input_length = input_encoding['input_ids'].shape[1]
        response_tokens = generated[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response_text.strip()

# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def train_faction_dialogue_model():
    """Main training script"""
    
    print("=== Angaraka AI Faction Dialogue Model Training ===")
    
    # Initialize components
    config = DialogueModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    data_path = "data/synthetic/dialogue"
    print("Loading training dataset...")
    train_dataset = FactionDialogueDataset(data_path, tokenizer, config.max_sequence_length)
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Initialize model
    model = FactionDialogueModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = DialogueTrainer(model, config)
    
    # Start training
    trainer.train(
        train_dataset=train_dataset,
        save_path="models/dialogue/faction_dialogue_model.pt"
    )
    
    print("Training completed! Model ready for ONNX export.")

if __name__ == "__main__":
    train_faction_dialogue_model()