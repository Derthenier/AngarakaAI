"""
Perfect Dialogue Model Trainer
Uses cleaned and enhanced data for high-quality faction responses
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import (
    AutoTokenizer, GPT2LMHeadModel, 
    get_linear_schedule_with_warmup, set_seed
)
from torch.optim import AdamW
import json
import os
import numpy as np
from typing import Dict, List
import sys

# Import path fix
sys.path.append(os.getcwd())

# Set random seeds for reproducibility
set_seed(42)

class PerfectDialogueConfig:
    """Optimized configuration for perfect dialogue training"""
    def __init__(self):
        self.base_model = "gpt2"
        self.max_sequence_length = 400  # Shorter for better quality
        
        # Optimized training parameters
        self.learning_rate = 3e-5  # Conservative learning rate
        self.warmup_ratio = 0.1
        self.max_epochs = 3  # More epochs with better data
        self.batch_size = 8  # Small batches for stability
        self.gradient_accumulation_steps = 2  # Effective batch size = 16
        self.weight_decay = 0.01
        
        # Generation parameters
        self.max_new_tokens = 80
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.05

class CleanDialogueDataset(Dataset):
    """Dataset that loads both cleaned and enhanced conversation data"""
    
    def __init__(self, tokenizer, max_length: int = 400):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Ensure tokenizer setup
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load both cleaned and enhanced data
        self._load_all_data()
        
        print(f"📊 Loaded {len(self.examples)} total training examples")
        self._print_data_statistics()
    
    def _load_all_data(self):
        """Load data from both cleaned and enhanced directories"""
        
        # Load cleaned original data
        cleaned_dir = "data/synthetic/dialogue_cleaned"
        if os.path.exists(cleaned_dir):
            print("📂 Loading cleaned original data...")
            self._load_data_from_directory(cleaned_dir, "cleaned")
        
        # Load enhanced high-quality data
        enhanced_dir = "data/synthetic/dialogue_enhanced" 
        if os.path.exists(enhanced_dir):
            print("📂 Loading enhanced training data...")
            self._load_data_from_directory(enhanced_dir, "enhanced")
    
    def _load_data_from_directory(self, directory: str, data_type: str):
        """Load conversation data from a directory"""
        
        files = [f for f in os.listdir(directory) if f.endswith('.json')]
        initial_count = len(self.examples)
        
        for file in files:
            file_path = os.path.join(directory, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for conversation in data['conversations']:
                self._process_conversation(conversation, data_type)
        
        added_count = len(self.examples) - initial_count
        print(f"  ✅ Added {added_count} examples from {data_type} data")
    
    def _process_conversation(self, conversation: Dict, data_type: str):
        """Process a conversation into training examples"""
        
        dialogue = conversation['dialogue']
        faction = conversation['npc_faction']
        relationship = conversation.get('relationship_context', 'neutral')
        
        # Extract player-NPC pairs
        for i in range(0, len(dialogue) - 1, 2):
            if (i < len(dialogue) - 1 and 
                dialogue[i]['speaker'] == 'player' and 
                dialogue[i + 1]['speaker'] == 'npc'):
                
                player_text = dialogue[i]['text'].strip()
                npc_text = dialogue[i + 1]['text'].strip()
                
                # Skip if either text is too short or contains artifacts
                if (len(player_text.split()) < 3 or 
                    len(npc_text.split()) < 5 or
                    self._contains_artifacts(npc_text)):
                    continue
                
                # Create clean training format
                # Format: Player: question Assistant: response<eos>
                input_prompt = f"[{faction.upper()}] Player: {player_text} Assistant:"
                full_text = f"{input_prompt} {npc_text}<|endoftext|>"
                
                self.examples.append({
                    'text': full_text,
                    'input_prompt': input_prompt,
                    'target_response': npc_text,
                    'faction': faction,
                    'relationship': relationship,
                    'data_type': data_type,
                    'input_length': len(self.tokenizer.encode(input_prompt))
                })
    
    def _contains_artifacts(self, text: str) -> bool:
        """Check if text contains problematic artifacts"""
        
        artifacts = [
            "your perspective, while",
            "misses the essential", 
            "...able,",
            "pc.",  # Corrupted tokens
            "wisdomable",
            "illusionable"
        ]
        
        text_lower = text.lower()
        return any(artifact in text_lower for artifact in artifacts)
    
    def _print_data_statistics(self):
        """Print statistics about the loaded data"""
        
        faction_counts = {}
        data_type_counts = {}
        
        for example in self.examples:
            faction = example['faction']
            data_type = example['data_type']
            
            faction_counts[faction] = faction_counts.get(faction, 0) + 1
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
        
        print("\n📈 Dataset Statistics:")
        print("By Faction:")
        for faction, count in sorted(faction_counts.items()):
            print(f"  {faction}: {count} examples")
        
        print("By Data Type:")
        for data_type, count in sorted(data_type_counts.items()):
            print(f"  {data_type}: {count} examples")
    
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
        
        # Create labels for language modeling
        labels = encoding['input_ids'].clone()
        
        # Mask input portion (only train on assistant response)
        input_length = example['input_length']
        if input_length < labels.shape[1]:
            labels[0, :input_length] = -100  # Ignore input tokens in loss
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class PerfectDialogueTrainer:
    """Enhanced trainer for high-quality dialogue generation"""
    
    def __init__(self, config: PerfectDialogueConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = GPT2LMHeadModel.from_pretrained(config.base_model)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"🚀 Trainer initialized on {self.device}")
        print(f"📝 Model: {config.base_model}")
        print(f"🔢 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, dataset: CleanDialogueDataset, save_path: str = "models/dialogue/perfect_faction_model.pt"):
        """Train the model with enhanced quality monitoring"""
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Calculate training steps
        total_steps = len(dataloader) * self.config.max_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\n🎯 Training Configuration:")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        
        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.max_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{self.config.max_epochs}")
            print("-" * 50)
            
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Progress logging
                if batch_idx % 25 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step:4d} | Batch {batch_idx:3d}/{len(dataloader)} | "
                          f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  🏆 New best loss! Saving model...")
                self.save_model(save_path.replace('.pt', '_best.pt'))
            
            # Save epoch checkpoint
            self.save_model(save_path.replace('.pt', f'_epoch_{epoch + 1}.pt'))
            
            # Test generation quality
            print(f"\n🧪 Testing generation quality...")
            self._test_generation_quality()
        
        # Save final model
        self.save_model(save_path)
        print(f"\n✅ Training completed! Final loss: {best_loss:.4f}")
        print(f"📁 Model saved to {save_path}")
    
    def _test_generation_quality(self):
        """Quick generation quality test during training"""
        
        self.model.eval()
        
        test_prompts = [
            "[ASHVATTHA] Player: What is karma? Assistant:",
            "[VAIKUNTHA] Player: Can karma be measured? Assistant:",
            "[YUGA_STRIDERS] Player: How do we break free? Assistant:",
            "[SHROUD_MANTRA] Player: What if reality is just a story? Assistant:"
        ]
        
        with torch.no_grad():
            for prompt in test_prompts:
                # Tokenize
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Generate
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=40,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                
                faction = prompt.split(']')[0][1:].lower()
                print(f"    {faction}: {response[:60]}...")
        
        self.model.train()
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'tokenizer_name': self.config.base_model
        }
        
        torch.save(checkpoint, path)

class PerfectDialogueInference:
    """Inference system for the perfectly trained model"""
    
    def __init__(self, model_path: str = "models/dialogue/perfect_faction_model_best.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config_dict = checkpoint['config']
        
        # Recreate config
        self.config = PerfectDialogueConfig()
        for key, value in config_dict.items():
            setattr(self.config, key, value)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint['tokenizer_name'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Perfect dialogue model loaded from {model_path}")
    
    def generate_response(self, player_input: str, faction: str, relationship: str = "neutral") -> str:
        """Generate high-quality faction response"""
        
        # Create input prompt
        prompt = f"[{faction.upper()}] Player: {player_input.strip()} Assistant:"
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate with careful parameters
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Clean up response
        if response.endswith('<|endoftext|>'):
            response = response[:-13].strip()
        
        return response

def train_perfect_model():
    """Main training function"""
    
    print("🌟 PERFECT FACTION DIALOGUE MODEL TRAINING")
    print("="*60)
    
    # Initialize components
    config = PerfectDialogueConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Load datasets
    print("📚 Loading training datasets...")
    dataset = CleanDialogueDataset(tokenizer, config.max_sequence_length)
    
    # Initialize trainer
    trainer = PerfectDialogueTrainer(config)
    
    # Train model
    trainer.train(dataset)
    
    print("\n🎉 Perfect model training completed!")

def test_perfect_model():
    """Test the perfectly trained model"""
    
    print("🧪 TESTING PERFECT MODEL")
    print("="*40)
    
    try:
        inference = PerfectDialogueInference()
        
        test_cases = [
            ("What is karma really?", "ashvattha"),
            ("Can karma be quantified?", "vaikuntha"),
            ("How do we escape suffering?", "yuga_striders"), 
            ("Is reality just a narrative?", "shroud_mantra"),
            ("Why should I follow dharma?", "ashvattha"),
            ("What's the optimal path?", "vaikuntha"),
            ("Is revolution necessary?", "yuga_striders"),
            ("Who controls the story?", "shroud_mantra")
        ]
        
        for question, faction in test_cases:
            response = inference.generate_response(question, faction)
            print(f"\n🎭 {faction.title()}")
            print(f"💬 Player: {question}")
            print(f"🤖 NPC: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")

if __name__ == "__main__":
    # Train the perfect model
    train_perfect_model()
    
    # Test after training
    print("\n" + "="*60)
    test_perfect_model()