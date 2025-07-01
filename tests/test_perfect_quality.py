"""
Quick Quality Test for Perfect Model
Test the trained model with proper attention mask handling
"""

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import sys
import os
sys.path.append(os.getcwd())

def test_perfect_model_quality():
    """Test the perfect model with proper attention mask handling"""
    
    print("🧪 TESTING PERFECT FACTION MODEL (Loss 0.0193)")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    try:
        checkpoint = torch.load("models/dialogue/perfect_faction_model_best.pt", map_location=device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Trying alternative model path...")
        
        try:
            checkpoint = torch.load("models/dialogue/perfect_faction_model.pt", map_location=device)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            print(f"✅ Model loaded from alternative path on {device}")
            
        except Exception as e2:
            print(f"❌ Failed to load from alternative path: {e2}")
            return
    
    # Test cases for each faction
    test_cases = [
        # Ashvattha - Ancient wisdom preservationists
        ("What is karma?", "ashvattha"),
        ("Why should I follow dharma?", "ashvattha"),
        ("Is reincarnation real?", "ashvattha"),
        
        # Vaikuntha - Algorithmic precision
        ("Can karma be measured?", "vaikuntha"),
        ("What's the optimal path?", "vaikuntha"),
        ("How do we quantify dharma?", "vaikuntha"),
        
        # Yuga Striders - Revolutionary liberation
        ("How do we break free?", "yuga_striders"),
        ("Is the system corrupt?", "yuga_striders"),
        ("Why destroy everything?", "yuga_striders"),
        
        # Shroud Mantra - Reality questioners
        ("Is reality just a story?", "shroud_mantra"),
        ("Who decides what's true?", "shroud_mantra"),
        ("What if nothing is real?", "shroud_mantra"),
    ]
    
    print(f"\n🎭 Testing {len(test_cases)} faction dialogue scenarios...")
    print()
    
    for question, faction in test_cases:
        print(f"🎭 {faction.upper()} FACTION")
        print(f"💬 Player: \"{question}\"")
        
        # Create prompt with proper formatting
        prompt = f"[{faction.upper()}] Player: {question} Assistant:"
        
        # Tokenize with proper attention mask
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=200
        ).to(device)
        
        # Generate response with proper attention mask
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # Proper attention mask
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the NPC response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        print(f"🤖 NPC Response: \"{response}\"")
        
        # Quick quality analysis
        response_lower = response.lower()
        
        # Check for faction-appropriate vocabulary
        faction_vocab_found = []
        if faction == "ashvattha":
            vocab = ["ancient", "sacred", "wisdom", "dharma", "eternal", "traditional"]
        elif faction == "vaikuntha":
            vocab = ["systematic", "optimal", "calculated", "analysis", "data", "algorithmic"]
        elif faction == "yuga_striders":
            vocab = ["liberation", "freedom", "break", "revolution", "chains", "destroy"]
        else:  # shroud_mantra
            vocab = ["narrative", "story", "perspective", "reality", "interpretation", "version"]
        
        for word in vocab:
            if word in response_lower:
                faction_vocab_found.append(word)
        
        # Check for philosophical concepts
        philosophy_words = ["karma", "dharma", "reincarnation", "consciousness", "existence", "truth"]
        philosophy_found = [word for word in philosophy_words if word in response_lower]
        
        # Quality indicators
        has_faction_vocab = len(faction_vocab_found) > 0
        has_philosophy = len(philosophy_found) > 0
        proper_length = 10 <= len(response.split()) <= 80
        no_artifacts = not any(x in response_lower for x in ["your perspective", "misses the essential", "...able"])
        
        print(f"📊 Quality Analysis:")
        print(f"   Faction vocabulary: {'✅' if has_faction_vocab else '❌'} {faction_vocab_found}")
        print(f"   Philosophical depth: {'✅' if has_philosophy else '❌'} {philosophy_found}")
        print(f"   Proper length: {'✅' if proper_length else '❌'} ({len(response.split())} words)")
        print(f"   Clean text: {'✅' if no_artifacts else '❌'}")
        
        overall_quality = "🌟 EXCELLENT" if all([has_faction_vocab, has_philosophy, proper_length, no_artifacts]) else \
                         "✅ GOOD" if sum([has_faction_vocab, has_philosophy, proper_length, no_artifacts]) >= 3 else \
                         "⚠️ NEEDS WORK"
        
        print(f"   Overall: {overall_quality}")
        print("-" * 60)
    
    print("\n🏆 TESTING COMPLETE!")
    print("With loss 0.0193, your model should show excellent faction personalities!")

if __name__ == "__main__":
    test_perfect_model_quality()