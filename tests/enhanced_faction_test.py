"""
Enhanced Faction Conditioning Fix
Strengthen faction differentiation in responses
"""

import torch
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.getcwd())

from models.dialogue.fixed_faction_model import FixedDialogueInference

class EnhancedFactionInference:
    """Enhanced inference with stronger faction conditioning"""
    
    def __init__(self, model_path: str = "models/dialogue/fixed_faction_model.pt"):
        self.base_inference = FixedDialogueInference(model_path)
        
        # Stronger faction prompts
        self.faction_prompts = {
            "ashvattha": "As a keeper of ancient wisdom and traditional dharma",
            "vaikuntha": "Using systematic analysis and algorithmic precision", 
            "yuga_striders": "From a revolutionary perspective seeking liberation",
            "shroud_mantra": "Considering multiple narrative interpretations"
        }
        
        # Faction-specific response starters
        self.faction_starters = {
            "ashvattha": [
                "The ancient scriptures teach us",
                "Sacred wisdom reveals", 
                "According to eternal dharma",
                "Traditional knowledge shows"
            ],
            "vaikuntha": [
                "Statistical analysis indicates",
                "Optimal calculations show",
                "Systematic evaluation proves", 
                "Data-driven assessment reveals"
            ],
            "yuga_striders": [
                "The truth they hide is",
                "Liberation requires understanding",
                "Break free from the illusion",
                "Revolutionary consciousness reveals"
            ],
            "shroud_mantra": [
                "From this perspective we see",
                "The narrative could be interpreted as",
                "Consider the alternative story",
                "Reality might be understood as"
            ]
        }
        
        # Filter words that shouldn't appear for each faction
        self.faction_filters = {
            "ashvattha": ["algorithmic", "systematic", "revolution", "narrative"],
            "vaikuntha": ["ancient", "sacred", "liberation", "story"],
            "yuga_striders": ["systematic", "traditional", "optimal", "interpretation"], 
            "shroud_mantra": ["sacred", "algorithmic", "liberation", "eternal"]
        }
    
    def generate_enhanced_response(self, player_input: str, faction: str, relationship: str) -> str:
        """Generate response with enhanced faction conditioning"""
        
        # Create stronger conditioning prompt
        faction_context = self.faction_prompts[faction]
        enhanced_input = f"{faction_context}, how would you respond to: {player_input}"
        
        # Format for the model
        formatted_input = f"[FACTION:{faction}] [REL:{relationship}] Player: {enhanced_input} Assistant:"
        
        # Get base response
        input_encoding = self.base_inference.tokenizer(
            formatted_input,
            return_tensors='pt'
        ).to(self.base_inference.device)
        
        # Generate multiple candidates and pick the best
        best_response = ""
        best_score = -1
        
        for attempt in range(3):  # Try 3 different responses
            with torch.no_grad():
                outputs = self.base_inference.model.generate_response(
                    input_ids=input_encoding['input_ids'],
                    attention_mask=input_encoding['attention_mask'],
                    max_new_tokens=80,
                    temperature=0.8 + (attempt * 0.1),  # Vary temperature
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.base_inference.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
            else:
                response = full_response[len(formatted_input):].strip()
            
            # Score this response for faction appropriateness
            score = self._score_faction_response(response, faction)
            
            if score > best_score:
                best_score = score
                best_response = response
        
        # Post-process the best response
        return self._post_process_response(best_response, faction)
    
    def _score_faction_response(self, response: str, faction: str) -> float:
        """Score how well response matches faction personality"""
        
        score = 0.0
        response_lower = response.lower()
        
        # Positive points for faction-appropriate vocabulary
        faction_vocab = {
            "ashvattha": ["ancient", "sacred", "wisdom", "dharma", "traditional", "eternal"],
            "vaikuntha": ["systematic", "optimal", "calculated", "analysis", "efficient", "data"],
            "yuga_striders": ["liberation", "freedom", "break", "revolution", "chains", "awakening"],
            "shroud_mantra": ["narrative", "perspective", "story", "interpretation", "reality", "version"]
        }
        
        vocab_words = faction_vocab.get(faction, [])
        for word in vocab_words:
            if word in response_lower:
                score += 1.0
        
        # Negative points for inappropriate vocabulary
        filter_words = self.faction_filters.get(faction, [])
        for word in filter_words:
            if word in response_lower:
                score -= 2.0
        
        # Negative points for repetitive templates
        if "your perspective, while" in response_lower:
            score -= 3.0
        
        if "misses the essential" in response_lower:
            score -= 3.0
        
        # Positive points for faction-specific response starters
        starters = self.faction_starters.get(faction, [])
        for starter in starters:
            if starter.lower() in response_lower:
                score += 2.0
        
        return score
    
    def _post_process_response(self, response: str, faction: str) -> str:
        """Clean up and enhance the response"""
        
        # Remove common problematic patterns
        response = response.replace("Your perspective, while", "")
        response = response.replace("misses the essential", "")
        response = response.replace("able,", "")
        
        # If response is too generic, prepend a faction-appropriate starter
        if len(response.strip()) < 20 or not any(word in response.lower() for word in ["karma", "dharma", "reincarnation", "consciousness"]):
            import random
            starter = random.choice(self.faction_starters[faction])
            response = f"{starter} that {response.strip()}"
        
        # Clean up spacing and grammar
        response = " ".join(response.split())  # Normalize whitespace
        response = response.strip()
        
        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?')):
            response += "."
        
        return response

def test_enhanced_conditioning():
    """Test the enhanced conditioning system"""
    
    print("=== Testing Enhanced Faction Conditioning ===")
    
    try:
        enhanced_inference = EnhancedFactionInference()
        
        test_scenarios = [
            ("What is karma?", "ashvattha", "neutral"),
            ("Is reincarnation real?", "vaikuntha", "hostile"), 
            ("How do we break free from suffering?", "yuga_striders", "trusting"),
            ("What if reality is just a story?", "shroud_mantra", "respectful"),
            ("Why should I follow dharma?", "ashvattha", "respectful"),
            ("Can karma be measured?", "vaikuntha", "neutral"),
            ("Is the system corrupt?", "yuga_striders", "allied"),
            ("Who decides what's true?", "shroud_mantra", "neutral")
        ]
        
        for player_input, faction, relationship in test_scenarios:
            print(f"\n🎭 {faction} ({relationship})")
            print(f"💬 Player: {player_input}")
            
            response = enhanced_inference.generate_enhanced_response(player_input, faction, relationship)
            print(f"🤖 Enhanced NPC: {response}")
            
            # Quick quality check
            response_lower = response.lower()
            has_faction_vocab = any(word in response_lower for word in ["karma", "dharma", "ancient", "systematic", "liberation", "narrative"])
            has_template = "your perspective" in response_lower or "misses the essential" in response_lower
            
            print(f"   📊 Quality: Faction vocab: {'✅' if has_faction_vocab else '❌'} | Template-free: {'✅' if not has_template else '❌'}")
        
    except Exception as e:
        print(f"❌ Enhanced testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Testing Enhanced Conditioning")
    test_enhanced_conditioning()