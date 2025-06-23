"""
Faction Dialogue Model Testing Script
Test the trained model with sample philosophical conversations
"""

import torch
from transformers import AutoTokenizer
import sys
import os

# Import path fix
sys.path.append(os.getcwd())

from models.dialogue.faction_dialogue_model import (
    FactionDialogueModel, DialogueModelConfig, DialogueInput, DialogueInference
)

class ModelTester:
    """Test trained faction dialogue model with sample inputs"""
    
    def __init__(self, model_path: str = "models/dialogue/faction_dialogue_model.pt"):
        print("Loading trained faction dialogue model...")
        
        # Initialize components
        self.config = DialogueModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load trained model
        self.model = FactionDialogueModel(self.config)
        self.load_model(model_path)
        self.model.to(self.device)
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
        
        print("✅ Model loaded successfully!")
    
    def load_model(self, path: str):
        """Load the trained model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {path}")
        except FileNotFoundError:
            print(f"❌ Model file not found: {path}")
            print("Make sure training completed and the model was saved.")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, conversation_history: list, player_input: str, 
                         faction: str, relationship: str) -> str:
        """Generate NPC response using the trained model"""
        
        # Format input text
        history_text = " ".join(conversation_history[-4:]) if conversation_history else ""
        input_text = f"Context: {history_text} Player: {player_input} NPC:"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=400,  # Leave room for response
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get faction and relationship IDs
        faction_id = torch.tensor([self.faction_to_id[faction]], device=self.device)
        relationship_id = torch.tensor([self.relationship_to_id[relationship]], device=self.device)
        
        # Generate response
        with torch.no_grad():
            # Use the model's generate_response method
            generated = self.model.generate_response(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                faction_id=faction_id,
                relationship_id=relationship_id,
                max_new_tokens=120,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode response (only the new tokens)
        input_length = input_encoding['input_ids'].shape[1]
        response_tokens = generated[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Clean up response
        response_text = response_text.strip()
        
        # Remove any remaining "NPC:" prefix if it appears
        if response_text.startswith("NPC:"):
            response_text = response_text[4:].strip()
        
        return response_text
    
    def run_comprehensive_tests(self):
        """Run a comprehensive test suite across all factions and scenarios"""
        
        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE FACTION DIALOGUE TESTING")
        print("="*60)
        
        # Test scenarios covering different philosophical topics and relationship dynamics
        test_scenarios = [
            {
                "name": "Karma Nature Discussion",
                "player_input": "What is karma, really? Is it just cause and effect?",
                "history": [],
                "topic": "karma_nature"
            },
            {
                "name": "Reincarnation Challenge", 
                "player_input": "What if reincarnation is just a story we tell ourselves to cope with death?",
                "history": ["NPC: The eternal cycle guides all existence.", "Player: But how can you be certain?"],
                "topic": "reincarnation_purpose"
            },
            {
                "name": "Cross-Faction Hostility",
                "player_input": "The other factions might have valid points too.",
                "history": ["NPC: Our path is the only true way.", "Player: But surely there are multiple truths?"],
                "topic": "dharma_interpretation"
            },
            {
                "name": "Deep Existential Question",
                "player_input": "What if consciousness itself creates reality rather than discovering it?",
                "history": ["NPC: Reality follows natural laws.", "Player: But what determines those laws?"],
                "topic": "consciousness_nature"
            },
            {
                "name": "Personal Doubt",
                "player_input": "Do you ever doubt your faction's teachings?",
                "history": ["NPC: Truth is absolute.", "Player: Even absolute truth can be questioned."],
                "topic": "free_will_fate"
            }
        ]
        
        factions = ["ashvattha", "vaikuntha", "yuga_striders", "shroud_mantra"]
        relationships = ["hostile", "neutral", "respectful", "trusting"]
        
        # Test each faction with each scenario
        for faction in factions:
            print(f"\n{'='*20} TESTING {faction.upper()} FACTION {'='*20}")
            
            for scenario in test_scenarios:
                print(f"\n📋 Scenario: {scenario['name']}")
                print(f"🎭 Faction: {faction}")
                print(f"💬 Player: \"{scenario['player_input']}\"")
                
                # Test different relationship levels for this scenario
                for relationship in relationships:
                    print(f"\n🤝 Relationship: {relationship}")
                    
                    try:
                        response = self.generate_response(
                            conversation_history=scenario['history'],
                            player_input=scenario['player_input'],
                            faction=faction,
                            relationship=relationship
                        )
                        
                        print(f"🤖 {faction.title()} NPC ({relationship}): \"{response}\"")
                        
                        # Basic quality checks
                        self._evaluate_response_quality(response, faction, relationship)
                        
                    except Exception as e:
                        print(f"❌ Error generating response: {e}")
                
                print("-" * 50)
    
    def _evaluate_response_quality(self, response: str, faction: str, relationship: str):
        """Basic evaluation of response quality"""
        
        quality_indicators = {
            "ashvattha": ["ancient", "sacred", "dharma", "wisdom", "traditional", "eternal"],
            "vaikuntha": ["optimal", "systematic", "calculated", "efficient", "data", "analysis"],
            "yuga_striders": ["liberation", "freedom", "break", "destruction", "revolution", "chains"],
            "shroud_mantra": ["narrative", "perspective", "story", "interpretation", "reality", "version"]
        }
        
        # Check for faction-specific vocabulary
        faction_words = quality_indicators.get(faction, [])
        faction_score = sum(1 for word in faction_words if word in response.lower())
        
        # Check response length (should be substantial)
        length_score = "✅" if 20 <= len(response.split()) <= 100 else "⚠️"
        
        # Check for philosophical depth
        philosophical_words = ["karma", "dharma", "reincarnation", "existence", "consciousness", "truth", "reality"]
        philosophy_score = sum(1 for word in philosophical_words if word in response.lower())
        
        print(f"   📊 Quality: Length {length_score} | Faction vocab: {faction_score} | Philosophy: {philosophy_score}")
        
        if faction_score == 0:
            print(f"   ⚠️  Warning: No faction-specific vocabulary detected")
        if philosophy_score == 0:
            print(f"   ⚠️  Warning: No philosophical concepts detected")
    
    def run_interactive_test(self):
        """Interactive testing mode for manual evaluation"""
        
        print("\n" + "="*60)
        print("🎮 INTERACTIVE TESTING MODE")
        print("="*60)
        print("Test your trained model with custom inputs!")
        print("Type 'quit' to exit, 'help' for commands")
        
        conversation_history = []
        current_faction = "ashvattha"
        current_relationship = "neutral"
        
        while True:
            print(f"\n🎭 Current: {current_faction} faction, {current_relationship} relationship")
            user_input = input("💬 Player input: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  faction <name>     - Change faction (ashvattha/vaikuntha/yuga_striders/shroud_mantra)")
                print("  relationship <rel> - Change relationship (hostile/suspicious/neutral/respectful/trusting/allied)")
                print("  clear             - Clear conversation history")
                print("  quit              - Exit interactive mode")
                continue
            elif user_input.lower().startswith('faction '):
                new_faction = user_input[8:].strip()
                if new_faction in self.faction_to_id:
                    current_faction = new_faction
                    print(f"✅ Changed to {current_faction} faction")
                else:
                    print("❌ Invalid faction. Use: ashvattha/vaikuntha/yuga_striders/shroud_mantra")
                continue
            elif user_input.lower().startswith('relationship '):
                new_relationship = user_input[13:].strip()
                if new_relationship in self.relationship_to_id:
                    current_relationship = new_relationship
                    print(f"✅ Changed to {current_relationship} relationship")
                else:
                    print("❌ Invalid relationship. Use: hostile/suspicious/neutral/respectful/trusting/allied")
                continue
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("✅ Conversation history cleared")
                continue
            
            # Generate response
            try:
                response = self.generate_response(
                    conversation_history=conversation_history,
                    player_input=user_input,
                    faction=current_faction,
                    relationship=current_relationship
                )
                
                print(f"🤖 {current_faction.title()} NPC: \"{response}\"")
                
                # Update conversation history
                conversation_history.append(f"Player: {user_input}")
                conversation_history.append(f"NPC: {response}")
                
                # Keep history manageable
                if len(conversation_history) > 8:
                    conversation_history = conversation_history[-8:]
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")

def main():
    """Main testing function"""
    
    try:
        # Initialize tester
        tester = ModelTester()
        
        print("\n🎯 Choose testing mode:")
        print("1. Comprehensive automated tests")
        print("2. Interactive testing mode")
        print("3. Both")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            tester.run_comprehensive_tests()
        
        if choice in ['2', '3']:
            tester.run_interactive_test()
        
        print("\n✅ Testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("Make sure your model training completed successfully.")

if __name__ == "__main__":
    main()