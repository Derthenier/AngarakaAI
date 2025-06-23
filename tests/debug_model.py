"""
Simple Model Debug Test
Isolate and fix the text generation issue
"""

import torch
from transformers import AutoTokenizer
import sys
import os

# Import path fix
sys.path.append(os.getcwd())

from models.dialogue.faction_dialogue_model import FactionDialogueModel, DialogueModelConfig

def debug_model_generation():
    """Debug the model's text generation step by step"""
    
    print("🔍 DEBUGGING MODEL GENERATION")
    print("="*50)
    
    # Load components
    config = DialogueModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded: {config.base_model}")
    print(f"✅ Vocab size: {len(tokenizer)}")
    
    # Test basic tokenization
    test_text = "Player: What is karma? NPC:"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\n📝 Tokenization Test:")
    print(f"Input: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")
    
    if decoded != test_text:
        print("⚠️ WARNING: Tokenization issue detected!")
    else:
        print("✅ Tokenization working correctly")
    
    # Load model
    try:
        model = FactionDialogueModel(config)
        checkpoint = torch.load("models/dialogue/faction_dialogue_model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test simple generation WITHOUT conditioning first
    print(f"\n🧪 Testing Basic Generation (No Conditioning)")
    
    input_text = "What is karma?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input: '{input_text}'")
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Simple forward pass
    with torch.no_grad():
        try:
            # Test base transformer only
            transformer_output = model.transformer(input_ids, attention_mask=attention_mask)
            print(f"✅ Base transformer works, hidden states shape: {transformer_output.last_hidden_state.shape}")
            
            # Test with dummy conditioning
            faction_id = torch.tensor([0], device=device)  # Ashvattha
            relationship_id = torch.tensor([2], device=device)  # Neutral
            
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                faction_id=faction_id,
                relationship_id=relationship_id
            )
            
            print(f"✅ Full model forward pass works, logits shape: {outputs['logits'].shape}")
            
            # Test simple greedy decoding
            logits = outputs['logits'][0, -1, :]  # Last token logits
            next_token_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            next_token_text = tokenizer.decode(next_token_id[0])
            
            print(f"Next token prediction: '{next_token_text}'")
            
            # Test generation with simpler parameters
            print(f"\n🎯 Testing Simple Generation")
            
            generated_ids = input_ids.clone()
            
            for step in range(20):  # Generate 20 tokens
                outputs = model.forward(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    faction_id=faction_id,
                    relationship_id=relationship_id
                )
                
                # Get next token (greedy decoding for debugging)
                next_token_logits = outputs['logits'][0, -1, :]
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                # Decode and check
                current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"Step {step+1}: '{current_text}'")
                
                # Stop if we hit end token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                # Stop if text gets too long
                if generated_ids.shape[1] > 100:
                    break
            
            final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response_only = final_text[len(input_text):].strip()
            
            print(f"\n📋 FINAL RESULT:")
            print(f"Full text: '{final_text}'")
            print(f"Response only: '{response_only}'")
            
            if len(response_only) > 0 and response_only.count(' ') > 2:
                print("✅ Model generating coherent text!")
            else:
                print("❌ Model not generating proper responses")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

def test_different_generation_methods():
    """Test different approaches to generation"""
    
    print(f"\n🔬 TESTING DIFFERENT GENERATION APPROACHES")
    print("="*50)
    
    # Try using transformers' built-in generation
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    try:
        # Test with basic GPT2 to verify our setup
        print("Testing with basic GPT2...")
        
        base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
        input_text = "What is karma?"
        input_ids = base_tokenizer.encode(input_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = base_model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True,
                pad_token_id=base_tokenizer.eos_token_id
            )
        
        generated_text = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        
        print(f"Basic GPT2 response: '{response}'")
        
        if len(response) > 10:
            print("✅ Basic generation works - issue is with our model")
        else:
            print("❌ Even basic generation has issues")
            
    except Exception as e:
        print(f"❌ Basic generation test failed: {e}")

if __name__ == "__main__":
    debug_model_generation()
    test_different_generation_methods()