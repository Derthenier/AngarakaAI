"""
Synthetic Dialogue Data Generator for Threads of Kaliyuga
Generates authentic philosophical conversations for each faction
"""

import random
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Import our framework modules
import sys
import os
sys.path.append(os.getcwd())

from lore.faction_profiles import get_faction_profile, FACTION_RELATIONSHIPS
from lore.philosophical_topics import (
    get_topic_by_name, TopicComplexity, CONVERSATION_STARTERS,
    get_faction_perspective, will_topic_generate_hostility
)
from models.dialogue.relationship_tracker import (
    RelationshipStatus, ConversationTone, TopicStance
)

@dataclass
class DialogueEntry:
    """A single dialogue exchange in a conversation"""
    speaker: str  # "player" or "npc"
    text: str
    emotion: str
    philosophical_depth: TopicComplexity
    faction_alignment_hints: List[str]  # Subtle clues about faction beliefs

@dataclass 
class ConversationDataset:
    """A complete conversation with metadata"""
    conversation_id: str
    npc_faction: str
    npc_name: str
    main_topic: str
    complexity_level: TopicComplexity
    relationship_context: RelationshipStatus
    
    # The actual conversation
    dialogue_exchanges: List[DialogueEntry]
    
    # Training metadata
    faction_personality_markers: List[str]  # Speech patterns exhibited
    philosophical_frameworks_used: List[str]  # Which philosophical concepts appeared
    hostility_triggers_activated: List[str]  # What made the NPC hostile/defensive
    relationship_impact: float  # How this conversation would affect relationship
    
    # Cross-faction dynamics
    other_factions_mentioned: List[str]
    cross_faction_hostility_demonstrated: bool

class SyntheticDialogueGenerator:
    """Generates philosophical dialogue datasets for faction training"""
    
    def __init__(self):
        # Template libraries for generating authentic dialogue
        self.player_question_templates = {
            TopicComplexity.SURFACE: [
                "I've been wondering about {topic}. What's your view on this?",
                "Your faction seems to believe {faction_view}. Can you explain why?",
                "How do you personally understand {concept}?",
                "What would you say to someone who disagrees with {faction_stance}?"
            ],
            TopicComplexity.MODERATE: [
                "If {philosophical_premise}, then how do you reconcile {contradiction}?",
                "I've heard arguments that {opposing_view}. How would you respond?", 
                "What if {challenging_scenario} - would that change your perspective?",
                "Some say your faction's approach to {topic} is {criticism}. Is that fair?"
            ],
            TopicComplexity.PROFOUND: [
                "What if {fundamental_question} - would that undermine everything you believe?",
                "I'm struggling with {existential_doubt}. How do you deal with such questions?",
                "If {core_assumption} is wrong, what happens to {belief_system}?",
                "Do you ever doubt {faction_certainty} when you see {contradictory_evidence}?"
            ],
            TopicComplexity.TRANSCENDENT: [
                "What if {reality_questioning} - how would we even know?",
                "Could it be that {consciousness_question} and we're all {illusion_suggestion}?",
                "If {existence_premise}, then what does that mean for {everything_we_know}?",
                "Sometimes I wonder if {ultimate_question} - do you ever feel that way?"
            ]
        }
        
        # NPC response framework by faction
        self.npc_response_frameworks = {
            "ashvattha": {
                "opening_phrases": [
                    "The ancient texts tell us...", "As our ancestors understood...",
                    "According to sacred wisdom...", "Traditional knowledge reveals...",
                    "The eternal dharma teaches..."
                ],
                "authority_appeals": [
                    "the sacred scriptures", "ancient sages", "timeless wisdom",
                    "dharmic tradition", "ancestral knowledge", "sacred teachings"
                ],
                "dismissal_phrases": [
                    "Modern thinking has corrupted...", "This contemporary confusion...",
                    "Such ideas stray from pure dharma...", "This misunderstands ancient truth..."
                ]
            },
            "vaikuntha": {
                "opening_phrases": [
                    "Statistical analysis shows...", "Optimal calculations indicate...",
                    "Systematic evaluation reveals...", "Data-driven assessment proves...",
                    "Algorithmic optimization demonstrates..."
                ],
                "authority_appeals": [
                    "empirical evidence", "mathematical precision", "systematic analysis",
                    "optimal algorithms", "calculated efficiency", "measured outcomes"
                ],
                "dismissal_phrases": [
                    "Such inefficient thinking...", "This lacks systematic rigor...",
                    "Emotional reasoning cannot...", "Unquantified beliefs are..."
                ]
            },
            "yuga_striders": {
                "opening_phrases": [
                    "Break free from the lie that...", "The truth they don't want you to know...",
                    "Liberation requires understanding...", "The system wants you to believe...",
                    "Shatter the illusion that..."
                ],
                "authority_appeals": [
                    "true freedom", "liberation from cycles", "breaking mental chains",
                    "revolutionary consciousness", "authentic choice", "pure awareness"
                ],
                "dismissal_phrases": [
                    "That's exactly what they want you to think...", "You're still trapped in...",
                    "Such thinking perpetuates the prison...", "This maintains the control system..."
                ]
            },
            "shroud_mantra": {
                "opening_phrases": [
                    "An interesting perspective, though...", "Reality is more flexible than...",
                    "Consider this alternative interpretation...", "The story could be told differently...",
                    "Truth depends on who's narrating..."
                ],
                "authority_appeals": [
                    "narrative flexibility", "interpretive possibilities", "perspective shifts",
                    "story reframing", "reality editing", "meaning creation"
                ],
                "dismissal_phrases": [
                    "Such rigid thinking...", "That's only one version of...",
                    "You're limiting yourself to...", "Reality is far more malleable than..."
                ]
            }
        }
    
    def generate_conversation(self, faction: str, topic: str, complexity: TopicComplexity,
                            relationship_status: RelationshipStatus, 
                            conversation_length: int = 8) -> ConversationDataset:
        """Generate a single philosophical conversation"""
        
        faction_profile = get_faction_profile(faction)
        topic_info = get_topic_by_name(topic)
        npc_name = self._generate_npc_name(faction)
        
        # Determine conversation dynamics
        hostility_level = will_topic_generate_hostility(topic, faction)
        conversation_tone = self._determine_conversation_tone(relationship_status, hostility_level)
        
        # Generate dialogue exchanges
        dialogue_exchanges = []
        
        # Opening player question
        opening_question = self._generate_player_question(topic, complexity, faction_profile)
        dialogue_exchanges.append(DialogueEntry(
            speaker="player",
            text=opening_question,
            emotion="curious",
            philosophical_depth=complexity,
            faction_alignment_hints=[]
        ))
        
        # Generate conversation flow
        for i in range(conversation_length - 1):
            if i % 2 == 0:  # NPC response
                npc_response = self._generate_npc_response(
                    faction, topic, complexity, conversation_tone, 
                    dialogue_exchanges[-1].text, i
                )
                dialogue_exchanges.append(npc_response)
            else:  # Player follow-up
                player_followup = self._generate_player_followup(
                    topic, complexity, dialogue_exchanges[-1].text, faction
                )
                dialogue_exchanges.append(player_followup)
        
        # Calculate metadata
        personality_markers = self._extract_personality_markers(dialogue_exchanges, faction)
        philosophical_frameworks = self._identify_philosophical_frameworks(dialogue_exchanges, topic)
        hostility_triggers = self._identify_hostility_triggers(dialogue_exchanges, faction, topic)
        relationship_impact = self._calculate_relationship_impact(dialogue_exchanges, faction, hostility_level)
        
        # Check for cross-faction mentions
        other_factions_mentioned, cross_faction_hostility = self._analyze_cross_faction_dynamics(dialogue_exchanges)
        
        return ConversationDataset(
            conversation_id=f"{faction}_{topic}_{complexity.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            npc_faction=faction,
            npc_name=npc_name,
            main_topic=topic,
            complexity_level=complexity,
            relationship_context=relationship_status,
            dialogue_exchanges=dialogue_exchanges,
            faction_personality_markers=personality_markers,
            philosophical_frameworks_used=philosophical_frameworks,
            hostility_triggers_activated=hostility_triggers,
            relationship_impact=relationship_impact,
            other_factions_mentioned=other_factions_mentioned,
            cross_faction_hostility_demonstrated=cross_faction_hostility
        )
    
    def _generate_npc_name(self, faction: str) -> str:
        """Generate appropriate NPC names for each faction"""
        names = {
            "ashvattha": ["Guru Ananda", "Sage Bhavani", "Acharya Devika", "Pandit Harish", "Master Kavitha"],
            "vaikuntha": ["Director Arjun", "Analyst Priya", "Coordinator Ravi", "Administrator Sneha", "System-Keeper Vikram"],
            "yuga_striders": ["Rebel Kiran", "Liberator Maya", "Revolutionary Tarun", "Breaker Zara", "Awakener Rohan"],
            "shroud_mantra": ["Narrator Asha", "Weaver Deepak", "Interpreter Nisha", "Editor Sameer", "Story-Keeper Lila"]
        }
        return random.choice(names.get(faction, ["Unknown"]))
    
    def _determine_conversation_tone(self, relationship: RelationshipStatus, 
                                   hostility_triggered: bool) -> ConversationTone:
        """Determine appropriate conversation tone"""
        if hostility_triggered:
            return ConversationTone.AGGRESSIVE if relationship == RelationshipStatus.HOSTILE else ConversationTone.DISMISSIVE
        
        tone_mapping = {
            RelationshipStatus.ALLIED: ConversationTone.INTIMATE,
            RelationshipStatus.TRUSTING: ConversationTone.FRIENDLY,
            RelationshipStatus.RESPECTFUL: ConversationTone.PHILOSOPHICAL,
            RelationshipStatus.NEUTRAL: ConversationTone.FORMAL,
            RelationshipStatus.SUSPICIOUS: ConversationTone.DISMISSIVE,
            RelationshipStatus.HOSTILE: ConversationTone.AGGRESSIVE
        }
        return tone_mapping.get(relationship, ConversationTone.FORMAL)
    
    def _generate_player_question(self, topic: str, complexity: TopicComplexity, 
                                faction_profile) -> str:
        """Generate an appropriate player question"""
        templates = self.player_question_templates[complexity]
        template = random.choice(templates)
        
        # Fill in template variables based on topic and faction
        topic_info = get_topic_by_name(topic)
        
        # Map faction profile names to topic perspective keys
        faction_mapping = {
            "Ashvattha Collective": "ashvattha",
            "Vaikuntha Initiative": "vaikuntha", 
            "Yuga Striders": "yuga_striders",
            "Shroud of Mantra": "shroud_mantra"
        }
        
        faction_key = faction_mapping.get(faction_profile.name, faction_profile.name.lower())
        faction_perspective = topic_info.faction_perspectives[faction_key]
        
        replacements = {
            "topic": topic_info.description,
            "faction_view": faction_perspective,
            "concept": random.choice(topic_info.subtopics),
            "faction_stance": faction_perspective,
            "philosophical_premise": random.choice(topic_info.central_questions),
            "contradiction": "suffering still exists despite karmic justice",
            "opposing_view": "karma might be entirely illusory",
            "challenging_scenario": "innocent people suffering greatly",
            "criticism": random.choice(["too rigid", "impractical", "harmful"]),
            "fundamental_question": random.choice(topic_info.central_questions),
            "existential_doubt": "whether any of this actually matters",
            "core_assumption": "reincarnation",
            "belief_system": "everything your faction teaches",
            "faction_certainty": faction_perspective.split()[0].lower(),
            "contradictory_evidence": "people living fulfilling lives without following dharma",
            "reality_questioning": "consciousness creates reality rather than discovering it",
            "consciousness_question": "individual awareness is an illusion",
            "illusion_suggestion": "fragments of a larger mind",
            "existence_premise": "existence itself is a story being written",
            "everything_we_know": "free will, karma, and dharma",
            "ultimate_question": "none of this is real"
        }
        
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def _generate_npc_response(self, faction: str, topic: str, complexity: TopicComplexity,
                             tone: ConversationTone, player_input: str, turn_number: int) -> DialogueEntry:
        """Generate NPC response based on faction personality"""
        
        faction_profile = get_faction_profile(faction)
        framework = self.npc_response_frameworks[faction]
        topic_info = get_topic_by_name(topic)
        
        # Build response components
        opening = random.choice(framework["opening_phrases"])
        authority = random.choice(framework["authority_appeals"])
        faction_perspective = topic_info.faction_perspectives[faction]  # faction is already the key
        
        # Generate response based on tone and complexity
        if tone == ConversationTone.AGGRESSIVE:
            dismissal = random.choice(framework["dismissal_phrases"])
            response_text = f"{dismissal} {faction_perspective.lower()}. {opening} {authority} proves this beyond question."
            emotion = "hostile"
        elif tone == ConversationTone.DISMISSIVE:
            response_text = f"{opening} {faction_perspective}. Your perspective, while understandable, misses the essential truth that {authority} reveals."
            emotion = "condescending"
        elif tone == ConversationTone.PHILOSOPHICAL:
            response_text = f"{opening} {faction_perspective}. Consider how {authority} illuminates this question in ways that transcend surface understanding."
            emotion = "contemplative"
        else:  # FORMAL, FRIENDLY, INTIMATE
            response_text = f"{opening} {faction_perspective}. Through {authority}, we understand that truth emerges when we align with this deeper wisdom."
            emotion = "earnest"
        
        # Add faction-specific vocabulary and speech patterns
        vocabulary = faction_profile.vocabulary_preferences
        response_text = self._enhance_with_faction_vocabulary(response_text, vocabulary, faction)
        
        # Add philosophical depth markers
        philosophical_hints = [
            f"faction_belief:{faction_perspective[:50]}",
            f"argumentation_style:{faction_profile.argumentation_style}",
            f"worldview:{faction_profile.tradition_stance}"
        ]
        
        return DialogueEntry(
            speaker="npc",
            text=response_text,
            emotion=emotion,
            philosophical_depth=complexity,
            faction_alignment_hints=philosophical_hints
        )
    
    def _generate_player_followup(self, topic: str, complexity: TopicComplexity, 
                                npc_response: str, faction: str) -> DialogueEntry:
        """Generate player follow-up questions"""
        
        followup_templates = [
            "But what about {counterargument}?",
            "I see your point, but {alternative_perspective}",
            "That's interesting, though {challenging_question}",
            "How do you reconcile that with {contradictory_evidence}?",
            "What would you say to someone who believes {opposing_view}?"
        ]
        
        template = random.choice(followup_templates)
        
        # Generate appropriate challenges based on faction
        if faction == "ashvattha":
            counterargs = ["modern discoveries that contradict ancient texts", 
                         "the suffering of those who follow dharma perfectly",
                         "the happiness of those who ignore tradition"]
        elif faction == "vaikuntha":
            counterargs = ["the unpredictability of human emotions",
                         "situations where optimal choices lead to suffering", 
                         "the value of inefficient but meaningful experiences"]
        elif faction == "yuga_striders":
            counterargs = ["people finding peace through acceptance of karma",
                         "the stability that some systems provide",
                         "gradual reform being more effective than destruction"]
        else:  # shroud_mantra
            counterargs = ["objective truths that exist regardless of perspective",
                         "the importance of having some stable beliefs",
                         "reality constraints that limit narrative flexibility"]
        
        counterargument = random.choice(counterargs)
        response_text = template.replace("{counterargument}", counterargument)
        response_text = response_text.replace("{alternative_perspective}", counterargument)
        response_text = response_text.replace("{challenging_question}", f"how do you address {counterargument}")
        response_text = response_text.replace("{contradictory_evidence}", counterargument)
        response_text = response_text.replace("{opposing_view}", counterargument)
        
        return DialogueEntry(
            speaker="player",
            text=response_text,
            emotion="questioning",
            philosophical_depth=complexity,
            faction_alignment_hints=[]
        )
    
    def _enhance_with_faction_vocabulary(self, text: str, vocabulary: List[str], faction: str) -> str:
        """Enhance text with faction-specific vocabulary"""
        # Simple vocabulary injection - replace generic terms with faction-specific ones
        replacements = {
            "ashvattha": {
                "understand": "comprehend through sacred wisdom",
                "believe": "know through ancient teaching",
                "think": "contemplate according to dharma",
                "truth": "eternal truth",
                "knowledge": "sacred knowledge"
            },
            "vaikuntha": {
                "understand": "calculate systematically", 
                "believe": "determine through analysis",
                "think": "process logically",
                "truth": "verified data",
                "knowledge": "quantified information"
            },
            "yuga_striders": {
                "understand": "see through the illusion",
                "believe": "awaken to the reality",
                "think": "break free from conditioning", 
                "truth": "liberating truth",
                "knowledge": "revolutionary awareness"
            },
            "shroud_mantra": {
                "understand": "interpret from this perspective",
                "believe": "choose to perceive",
                "think": "construct meaning around",
                "truth": "chosen narrative",
                "knowledge": "crafted understanding"
            }
        }
        
        if faction in replacements:
            for generic, specific in replacements[faction].items():
                text = text.replace(generic, specific)
        
        return text
    
    def _extract_personality_markers(self, exchanges: List[DialogueEntry], faction: str) -> List[str]:
        """Extract faction personality markers from dialogue"""
        faction_profile = get_faction_profile(faction)
        markers = []
        
        for exchange in exchanges:
            if exchange.speaker == "npc":
                # Check for speech patterns
                for pattern in faction_profile.speech_patterns:
                    if pattern in exchange.text.lower() or any(word in exchange.text.lower() 
                                                             for word in pattern.split("_")):
                        markers.append(f"speech_pattern:{pattern}")
                
                # Check for vocabulary usage
                for vocab in faction_profile.vocabulary_preferences:
                    if vocab in exchange.text.lower():
                        markers.append(f"vocabulary:{vocab}")
        
        return list(set(markers))  # Remove duplicates
    
    def _identify_philosophical_frameworks(self, exchanges: List[DialogueEntry], topic: str) -> List[str]:
        """Identify which philosophical frameworks were referenced"""
        topic_info = get_topic_by_name(topic)
        frameworks = []
        
        for exchange in exchanges:
            for framework in topic_info.philosophical_frameworks:
                if any(keyword in exchange.text.lower() for keyword in framework.split("_")):
                    frameworks.append(framework)
        
        return list(set(frameworks))
    
    def _identify_hostility_triggers(self, exchanges: List[DialogueEntry], faction: str, topic: str) -> List[str]:
        """Identify what triggered hostility in the conversation"""
        triggers = []
        
        for exchange in exchanges:
            if exchange.emotion in ["hostile", "condescending", "defensive"]:
                triggers.append(f"topic:{topic}")
                if "faction" in exchange.text.lower():
                    triggers.append("faction_criticism")
                if any(word in exchange.text.lower() for word in ["wrong", "false", "mistaken"]):
                    triggers.append("direct_contradiction")
        
        return triggers
    
    def _calculate_relationship_impact(self, exchanges: List[DialogueEntry], faction: str, hostility: bool) -> float:
        """Calculate how this conversation affects the relationship"""
        base_impact = 0.0
        
        for exchange in exchanges:
            if exchange.speaker == "player":
                if exchange.emotion == "questioning":
                    base_impact += 0.1  # Thoughtful questions improve relationship
                elif exchange.emotion == "challenging":
                    base_impact -= 0.2  # Challenges may damage relationship
            elif exchange.speaker == "npc":
                if exchange.emotion == "hostile":
                    base_impact -= 0.3
                elif exchange.emotion == "contemplative":
                    base_impact += 0.2
        
        # Hostility modifier
        if hostility:
            base_impact -= 0.5
        
        return max(-1.0, min(1.0, base_impact))
    
    def _analyze_cross_faction_dynamics(self, exchanges: List[DialogueEntry]) -> Tuple[List[str], bool]:
        """Analyze mentions of other factions and hostility toward them"""
        factions_mentioned = []
        hostility_demonstrated = False
        
        faction_keywords = {
            "ashvattha": ["traditional", "ancient", "sacred", "preservation"],
            "vaikuntha": ["systematic", "algorithmic", "optimization", "calculation"],
            "yuga_striders": ["destruction", "liberation", "revolution", "breaking"],
            "shroud_mantra": ["narrative", "perspective", "interpretation", "story"]
        }
        
        for exchange in exchanges:
            text_lower = exchange.text.lower()
            for faction, keywords in faction_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    if faction not in factions_mentioned:
                        factions_mentioned.append(faction)
                    
                    # Check for hostile language
                    if any(hostile_word in text_lower for hostile_word in 
                           ["corrupt", "wrong", "foolish", "dangerous", "destructive"]):
                        hostility_demonstrated = True
        
        return factions_mentioned, hostility_demonstrated
    
    def generate_faction_dataset(self, faction: str, num_conversations: int = 400) -> List[ConversationDataset]:
        """Generate a complete dataset for one faction"""
        
        conversations = []
        topics = ["karma_nature", "reincarnation_purpose", "dharma_interpretation", 
                 "free_will_fate", "consciousness_nature"]
        complexities = list(TopicComplexity)
        relationships = list(RelationshipStatus)
        
        print(f"Generating {num_conversations} conversations for {faction}...")
        
        for i in range(num_conversations):
            # Random selection with some weighting toward profound discussions
            topic = random.choice(topics)
            complexity = random.choices(
                complexities, 
                weights=[10, 20, 40, 30],  # Weight toward moderate/profound
                k=1
            )[0]
            relationship = random.choice(relationships)
            
            conversation = self.generate_conversation(
                faction, topic, complexity, relationship,
                conversation_length=random.randint(6, 12)
            )
            conversations.append(conversation)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_conversations} conversations...")
        
        print(f"Completed {faction} dataset!")
        return conversations
    
    def save_dataset(self, conversations: List[ConversationDataset], output_path: str):
        """Save conversation dataset to JSON"""
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "generator_version": "1.0"
            },
            "conversations": []
        }
        
        for conv in conversations:
            conv_data = {
                "id": conv.conversation_id,
                "npc_faction": conv.npc_faction,
                "npc_name": conv.npc_name,
                "main_topic": conv.main_topic,
                "complexity_level": conv.complexity_level.value,
                "relationship_context": conv.relationship_context.value,
                "relationship_impact": conv.relationship_impact,
                "dialogue": [
                    {
                        "speaker": exchange.speaker,
                        "text": exchange.text,
                        "emotion": exchange.emotion,
                        "philosophical_depth": exchange.philosophical_depth.value,
                        "faction_alignment_hints": exchange.faction_alignment_hints
                    }
                    for exchange in conv.dialogue_exchanges
                ],
                "training_metadata": {
                    "personality_markers": conv.faction_personality_markers,
                    "philosophical_frameworks": conv.philosophical_frameworks_used,
                    "hostility_triggers": conv.hostility_triggers_activated,
                    "cross_faction_mentions": conv.other_factions_mentioned,
                    "cross_faction_hostility": conv.cross_faction_hostility_demonstrated
                }
            }
            dataset["conversations"].append(conv_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_path}")

# ============================================================================
# MAIN GENERATION SCRIPT
# ============================================================================

def generate_all_faction_datasets(conversations_per_faction: int = 400):
    """Generate complete datasets for all factions"""
    
    generator = SyntheticDialogueGenerator()
    factions = ["ashvattha", "vaikuntha", "yuga_striders", "shroud_mantra"]
    
    print("=== Angaraka AI Dialogue Dataset Generation ===")
    print(f"Generating {conversations_per_faction} conversations per faction...")
    print()
    
    for faction in factions:
        print(f"Starting {faction} faction dataset...")
        conversations = generator.generate_faction_dataset(faction, conversations_per_faction)
        
        # Save individual faction dataset
        output_path = f"data/synthetic/dialogue/{faction}_conversations.json"
        generator.save_dataset(conversations, output_path)
        print()
    
    print("=== Dataset Generation Complete! ===")
    print(f"Generated datasets for all factions in data/synthetic/dialogue/")
    print("Ready for model training!")

if __name__ == "__main__":
    # Generate datasets - adjust conversation count as needed
    generate_all_faction_datasets(conversations_per_faction=300)