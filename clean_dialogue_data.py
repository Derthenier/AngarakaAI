"""
Clean Data Reprocessor for Perfect Dialogue
Fixes training data quality issues for coherent faction responses
"""

import json
import os
import re
from typing import Dict, List, Tuple

import sys
sys.path.append(os.getcwd())

from lore.faction_profiles import get_faction_profile
from lore.philosophical_topics import get_topic_by_name

class DialogueDataCleaner:
    """Clean and improve dialogue training data quality"""
    
    def __init__(self):
        self.faction_vocabulary = {
            "ashvattha": {
                "core_words": ["ancient", "sacred", "eternal", "dharma", "wisdom", "traditional", "ancestors"],
                "speech_patterns": ["The ancient texts tell us", "Sacred wisdom reveals", "Eternal dharma teaches"],
                "forbidden_words": ["algorithmic", "systematic", "liberation", "narrative", "data", "optimization"]
            },
            "vaikuntha": {
                "core_words": ["systematic", "optimal", "calculated", "efficient", "analysis", "data", "algorithmic"],
                "speech_patterns": ["Statistical analysis shows", "Optimal calculations indicate", "Data reveals"],
                "forbidden_words": ["ancient", "sacred", "liberation", "narrative", "ancestors", "eternal"]
            },
            "yuga_striders": {
                "core_words": ["liberation", "freedom", "revolution", "break", "chains", "awakening", "destroy"],
                "speech_patterns": ["Break free from", "Liberation requires", "The truth they hide"],
                "forbidden_words": ["systematic", "optimal", "sacred", "ancient", "calculated", "narrative"]
            },
            "shroud_mantra": {
                "core_words": ["narrative", "story", "perspective", "interpretation", "reality", "version", "meaning"],
                "speech_patterns": ["Consider this perspective", "The story could be", "Reality might be"],
                "forbidden_words": ["sacred", "systematic", "liberation", "ancient", "optimal", "eternal"]
            }
        }
        
        # High-quality response templates for each faction
        self.faction_templates = {
            "ashvattha": [
                "The ancient scriptures teach us that {concept}. Through sacred wisdom, we understand {explanation}.",
                "According to eternal dharma, {concept} represents {explanation}. This truth has guided souls for millennia.",
                "Traditional knowledge reveals that {concept}. Our ancestors preserved this wisdom: {explanation}."
            ],
            "vaikuntha": [
                "Statistical analysis demonstrates that {concept} can be quantified as {explanation}. Optimal outcomes require this understanding.",
                "Systematic evaluation proves {concept} operates through {explanation}. Calculated precision yields truth.",
                "Data-driven assessment shows {concept}. Through algorithmic analysis, we determine {explanation}."
            ],
            "yuga_striders": [
                "The liberating truth is that {concept} represents {explanation}. Break free from this deception!",
                "Revolutionary consciousness reveals {concept} as {explanation}. Shatter these mental chains!",
                "Awakening requires understanding that {concept}. True freedom means {explanation}."
            ],
            "shroud_mantra": [
                "From this perspective, {concept} could be interpreted as {explanation}. Reality is more flexible than it appears.",
                "Consider the narrative where {concept} represents {explanation}. Truth depends on who tells the story.",
                "The story of {concept} might be understood as {explanation}. Meaning shifts with perspective."
            ]
        }
    
    def clean_existing_conversations(self, input_dir: str, output_dir: str):
        """Clean and improve existing conversation data"""
        
        print("🧹 CLEANING EXISTING CONVERSATION DATA")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        faction_files = [
            "ashvattha_conversations.json",
            "vaikuntha_conversations.json", 
            "yuga_striders_conversations.json",
            "shroud_mantra_conversations.json"
        ]
        
        for faction_file in faction_files:
            faction_name = faction_file.rsplit('_', 1)[0]
            input_path = os.path.join(input_dir, faction_file)
            output_path = os.path.join(output_dir, f"clean_{faction_file}")
            
            if os.path.exists(input_path):
                print(f"\n📂 Processing {faction_name}...")
                self._clean_faction_file(input_path, output_path, faction_name)
    
    def _clean_faction_file(self, input_path: str, output_path: str, faction: str):
        """Clean a single faction's conversation file"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_conversations = []
        stats = {"total": 0, "cleaned": 0, "rejected": 0}
        
        for conversation in data['conversations']:
            stats["total"] += 1
            
            cleaned_conv = self._clean_conversation(conversation, faction)
            if cleaned_conv:
                cleaned_conversations.append(cleaned_conv)
                stats["cleaned"] += 1
            else:
                stats["rejected"] += 1
        
        # Save cleaned data
        cleaned_data = {
            "metadata": {
                "original_file": input_path,
                "faction": faction,
                "cleaning_stats": stats,
                "cleaned_at": "2024-01-01"
            },
            "conversations": cleaned_conversations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ {faction}: {stats['cleaned']}/{stats['total']} conversations cleaned")
        print(f"     Rejection rate: {stats['rejected']/stats['total']*100:.1f}%")
    
    def _clean_conversation(self, conversation: Dict, faction: str) -> Dict:
        """Clean a single conversation"""
        
        dialogue = conversation['dialogue']
        cleaned_dialogue = []
        
        # Process dialogue exchanges
        for i, exchange in enumerate(dialogue):
            if exchange['speaker'] == 'npc':
                original_text = exchange['text']
                cleaned_text = self._clean_npc_response(original_text, faction)
                
                # Reject if cleaning failed
                if not cleaned_text or len(cleaned_text.split()) < 5:
                    return None
                
                # Update exchange
                cleaned_exchange = exchange.copy()
                cleaned_exchange['text'] = cleaned_text
                cleaned_dialogue.append(cleaned_exchange)
            else:
                # Keep player exchanges as-is (mostly)
                cleaned_dialogue.append(exchange)
        
        # Only keep conversations with substantial content
        if len(cleaned_dialogue) < 4:  # At least 2 exchanges
            return None
        
        # Create cleaned conversation
        cleaned_conv = conversation.copy()
        cleaned_conv['dialogue'] = cleaned_dialogue
        
        return cleaned_conv
    
    def _clean_npc_response(self, text: str, faction: str) -> str:
        """Clean a single NPC response"""
        
        # Remove problematic patterns
        text = re.sub(r'Your perspective, while.*?able,', '', text)
        text = re.sub(r'misses the essential.*?truth that', '', text)
        text = re.sub(r'\.+', '.', text)  # Multiple periods
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        
        # Remove sentence fragments
        sentences = text.split('.')
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 4:  # At least 4 words
                valid_sentences.append(sentence)
        
        if not valid_sentences:
            return ""
        
        cleaned_text = '. '.join(valid_sentences) + '.'
        
        # Check faction appropriateness
        if not self._is_faction_appropriate(cleaned_text, faction):
            # Try to fix with faction-specific response
            return self._generate_faction_response(faction)
        
        # Enhance with faction vocabulary if weak
        if self._needs_faction_enhancement(cleaned_text, faction):
            cleaned_text = self._enhance_with_faction_vocabulary(cleaned_text, faction)
        
        return cleaned_text
    
    def _is_faction_appropriate(self, text: str, faction: str) -> bool:
        """Check if response is appropriate for the faction"""
        
        text_lower = text.lower()
        faction_vocab = self.faction_vocabulary[faction]
        
        # Check for forbidden words
        for word in faction_vocab["forbidden_words"]:
            if word in text_lower:
                return False
        
        # Check for required faction vocabulary
        core_words_found = sum(1 for word in faction_vocab["core_words"] if word in text_lower)
        
        return core_words_found >= 1  # At least one faction word
    
    def _needs_faction_enhancement(self, text: str, faction: str) -> bool:
        """Check if response needs more faction-specific vocabulary"""
        
        text_lower = text.lower()
        faction_vocab = self.faction_vocabulary[faction]
        
        core_words_found = sum(1 for word in faction_vocab["core_words"] if word in text_lower)
        
        return core_words_found < 2  # Needs more faction words
    
    def _enhance_with_faction_vocabulary(self, text: str, faction: str) -> str:
        """Enhance response with faction-specific vocabulary"""
        
        faction_vocab = self.faction_vocabulary[faction]
        
        # Add faction-appropriate starter if needed
        if not any(pattern.lower() in text.lower() for pattern in faction_vocab["speech_patterns"]):
            import random
            starter = random.choice(faction_vocab["speech_patterns"])
            text = f"{starter} that {text.lower()}"
        
        return text
    
    def _generate_faction_response(self, faction: str) -> str:
        """Generate a clean faction-appropriate response"""
        
        import random
        
        templates = self.faction_templates[faction]
        template = random.choice(templates)
        
        # Fill template with appropriate content
        if faction == "ashvattha":
            concept = "karma"
            explanation = "the eternal law governing all actions and consequences"
        elif faction == "vaikuntha":
            concept = "karma"
            explanation = "a measurable energy system that can be optimized"
        elif faction == "yuga_striders":
            concept = "karma"
            explanation = "a control mechanism designed to perpetuate suffering"
        else:  # shroud_mantra
            concept = "karma" 
            explanation = "a narrative we tell ourselves about cause and effect"
        
        return template.format(concept=concept, explanation=explanation)
    
    def create_enhanced_training_examples(self, output_dir: str, examples_per_faction: int = 100):
        """Create additional high-quality training examples"""
        
        print(f"\n🎯 CREATING {examples_per_faction} ENHANCED EXAMPLES PER FACTION")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        philosophical_topics = [
            ("karma", ["What is karma?", "How does karma work?", "Is karma fair?"]),
            ("dharma", ["What is dharma?", "How do I find my dharma?", "Is dharma universal?"]),
            ("reincarnation", ["Is reincarnation real?", "Why don't we remember past lives?", "Can reincarnation be proven?"]),
            ("consciousness", ["What is consciousness?", "Is consciousness fundamental?", "Can consciousness survive death?"]),
            ("reality", ["What is reality?", "Is reality objective?", "Who determines reality?"])
        ]
        
        factions = ["ashvattha", "vaikuntha", "yuga_striders", "shroud_mantra"]
        
        for faction in factions:
            print(f"\n📝 Creating examples for {faction}...")
            
            enhanced_conversations = []
            
            for i in range(examples_per_faction):
                # Pick random topic and question
                topic, questions = philosophical_topics[i % len(philosophical_topics)]
                question = questions[i % len(questions)]
                
                # Generate high-quality response
                response = self._create_perfect_response(question, topic, faction)
                
                # Create conversation structure
                conversation = {
                    "id": f"enhanced_{faction}_{i:03d}",
                    "npc_faction": faction,
                    "npc_name": f"Enhanced_{faction.title()}",
                    "main_topic": topic,
                    "complexity_level": "moderate",
                    "relationship_context": "neutral",
                    "dialogue": [
                        {
                            "speaker": "player",
                            "text": question,
                            "emotion": "curious"
                        },
                        {
                            "speaker": "npc", 
                            "text": response,
                            "emotion": "earnest"
                        }
                    ]
                }
                
                enhanced_conversations.append(conversation)
            
            # Save enhanced examples
            output_file = os.path.join(output_dir, f"enhanced_{faction}_conversations.json")
            enhanced_data = {
                "metadata": {
                    "type": "enhanced_training_data",
                    "faction": faction,
                    "count": len(enhanced_conversations)
                },
                "conversations": enhanced_conversations
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Created {len(enhanced_conversations)} enhanced examples")
    
    def _create_perfect_response(self, question: str, topic: str, faction: str) -> str:
        """Create a perfect faction-appropriate response"""
        
        faction_profile = get_faction_profile(faction)
        
        # Get faction's perspective on the topic
        if topic == "karma":
            perspective = faction_profile.karma_belief
        elif topic == "dharma":
            perspective = faction_profile.dharma_interpretation
        elif topic == "reincarnation":
            perspective = faction_profile.reincarnation_view
        elif topic == "consciousness":
            if faction == "ashvattha":
                perspective = "Sacred essence connecting souls to cosmic dharma through eternal wisdom"
            elif faction == "vaikuntha":
                perspective = "Emergent phenomenon that can be optimized through systematic enhancement"
            elif faction == "yuga_striders":
                perspective = "Imprisoned awareness that must be liberated from all binding systems"
            else:
                perspective = "Reality-creating force that shapes existence through perception and belief"
        else:  # reality
            if faction == "ashvattha":
                perspective = "Eternal truth revealed through sacred texts and ancient wisdom"
            elif faction == "vaikuntha":
                perspective = "Quantifiable system that can be measured and optimized"
            elif faction == "yuga_striders":
                perspective = "Constructed illusion designed to maintain control over consciousness"
            else:
                perspective = "Malleable narrative that can be rewritten and reinterpreted"
        
        # Create response using faction template
        import random
        template = random.choice(self.faction_templates[faction])
        
        response = template.format(
            concept=topic,
            explanation=perspective.lower()
        )
        
        return response

def main():
    """Run the complete data cleaning and enhancement process"""
    
    print("🚀 DIALOGUE DATA CLEANING & ENHANCEMENT")
    print("="*60)
    
    cleaner = DialogueDataCleaner()
    
    # Step 1: Clean existing data
    print("\n📋 Step 1: Cleaning existing conversation data...")
    cleaner.clean_existing_conversations(
        input_dir="data/synthetic/dialogue",
        output_dir="data/synthetic/dialogue_cleaned"
    )
    
    # Step 2: Create enhanced examples
    print("\n📋 Step 2: Creating enhanced training examples...")
    cleaner.create_enhanced_training_examples(
        output_dir="data/synthetic/dialogue_enhanced",
        examples_per_faction=150  # High-quality examples
    )
    
    print("\n✅ DATA CLEANING COMPLETE!")
    print("\nNext steps:")
    print("1. Review cleaned data in data/synthetic/dialogue_cleaned/")
    print("2. Check enhanced examples in data/synthetic/dialogue_enhanced/")
    print("3. Retrain model with combined clean + enhanced data")

if __name__ == "__main__":
    main()