"""
Relationship Tracking System for Threads of Kaliyuga
Maintains conversation continuity and emotional development across interactions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import json
from datetime import datetime, timedelta

class RelationshipStatus(Enum):
    """Overall relationship status between NPC and player"""
    HOSTILE = "hostile"
    SUSPICIOUS = "suspicious" 
    NEUTRAL = "neutral"
    RESPECTFUL = "respectful"
    TRUSTING = "trusting"
    ALLIED = "allied"

class ConversationTone(Enum):
    """Tone of most recent conversation"""
    AGGRESSIVE = "aggressive"
    DISMISSIVE = "dismissive"
    FORMAL = "formal" 
    FRIENDLY = "friendly"
    PHILOSOPHICAL = "philosophical"
    INTIMATE = "intimate"

class TopicStance(Enum):
    """NPC's stance on player's views on specific topics"""
    STRONGLY_OPPOSES = "strongly_opposes"
    OPPOSES = "opposes"
    SKEPTICAL = "skeptical"
    NEUTRAL = "neutral"
    INTERESTED = "interested" 
    AGREES = "agrees"
    STRONGLY_AGREES = "strongly_agrees"

@dataclass
class ConversationMemory:
    """Memory of a specific conversation"""
    timestamp: datetime
    topic_discussed: str
    player_stance: str
    npc_response_tone: ConversationTone
    relationship_impact: float  # -1.0 to 1.0
    memorable_quotes: List[str] = field(default_factory=list)
    philosophical_points_made: List[str] = field(default_factory=list)

@dataclass
class TopicHistory:
    """NPC's memory and stance development on a specific topic"""
    topic_name: str
    times_discussed: int
    current_stance: TopicStance
    stance_evolution: List[TopicStance] = field(default_factory=list)
    key_arguments_heard: List[str] = field(default_factory=list)
    npc_counterarguments_used: List[str] = field(default_factory=list)
    
    def record_discussion(self, player_argument: str, npc_response: str, stance_change: Optional[TopicStance] = None):
        """Record a discussion about this topic"""
        self.times_discussed += 1
        self.key_arguments_heard.append(player_argument)
        self.npc_counterarguments_used.append(npc_response)
        
        if stance_change and stance_change != self.current_stance:
            self.stance_evolution.append(self.current_stance)
            self.current_stance = stance_change

@dataclass 
class NPCRelationshipProfile:
    """Complete relationship profile for an NPC with the player"""
    npc_id: str
    npc_name: str
    faction: str
    
    # Current relationship state
    relationship_status: RelationshipStatus
    trust_level: float  # 0.0 to 1.0
    respect_level: float  # 0.0 to 1.0
    philosophical_alignment: float  # -1.0 to 1.0 (opposing to aligned)
    
    # Conversation history
    conversation_memories: List[ConversationMemory] = field(default_factory=list)
    topic_histories: Dict[str, TopicHistory] = field(default_factory=dict)
    
    # Personal knowledge about player
    known_player_beliefs: Dict[str, str] = field(default_factory=dict)
    player_faction_alignment_perceived: Optional[str] = None
    personal_secrets_shared: List[str] = field(default_factory=list)
    
    # Behavioral changes
    speech_patterns_adapted: List[str] = field(default_factory=list)
    topics_to_avoid: Set[str] = field(default_factory=set)
    topics_of_interest: Set[str] = field(default_factory=set)
    
    # Temporal factors
    last_interaction: Optional[datetime] = None
    relationship_decay_rate: float = 0.1  # How fast relationship degrades without interaction
    
    def add_conversation_memory(self, topic: str, player_stance: str, npc_tone: ConversationTone, 
                               impact: float, quotes: List[str] = None, points: List[str] = None):
        """Add a new conversation to memory"""
        memory = ConversationMemory(
            timestamp=datetime.now(),
            topic_discussed=topic,
            player_stance=player_stance,
            npc_response_tone=npc_tone,
            relationship_impact=impact,
            memorable_quotes=quotes or [],
            philosophical_points_made=points or []
        )
        self.conversation_memories.append(memory)
        self.last_interaction = datetime.now()
        
        # Update relationship metrics
        self.update_relationship_metrics(impact)
        
        # Update topic history
        if topic not in self.topic_histories:
            self.topic_histories[topic] = TopicHistory(
                topic_name=topic,
                times_discussed=0,
                current_stance=TopicStance.NEUTRAL
            )
    
    def update_relationship_metrics(self, impact: float):
        """Update trust, respect, and alignment based on conversation impact"""
        # Impact affects different metrics differently based on faction
        if self.faction == "ashvattha":
            # Values respect for tradition and wisdom
            if impact > 0:
                self.respect_level = min(1.0, self.respect_level + abs(impact) * 0.8)
                self.trust_level = min(1.0, self.trust_level + abs(impact) * 0.5)
            else:
                self.respect_level = max(0.0, self.respect_level + impact * 0.9)
                
        elif self.faction == "vaikuntha":
            # Values logical consistency and order
            if impact > 0:
                self.philosophical_alignment = min(1.0, self.philosophical_alignment + abs(impact) * 0.7)
                self.respect_level = min(1.0, self.respect_level + abs(impact) * 0.6)
            else:
                self.philosophical_alignment = max(-1.0, self.philosophical_alignment + impact * 0.8)
                
        elif self.faction == "yuga_striders":
            # Values passion and revolutionary thinking
            if impact > 0:
                self.trust_level = min(1.0, self.trust_level + abs(impact) * 0.9)
                self.philosophical_alignment = min(1.0, self.philosophical_alignment + abs(impact) * 0.8)
            else:
                self.trust_level = max(0.0, self.trust_level + impact * 1.0)
                
        elif self.faction == "shroud_mantra":
            # Values intellectual flexibility and questioning
            if impact > 0:
                self.respect_level = min(1.0, self.respect_level + abs(impact) * 0.7)
                self.philosophical_alignment = min(1.0, self.philosophical_alignment + abs(impact) * 0.5)
            else:
                # Shroud members are less affected by disagreement
                self.respect_level = max(0.0, self.respect_level + impact * 0.4)
        
        # Update overall relationship status
        self.update_relationship_status()
    
    def update_relationship_status(self):
        """Update relationship status based on current metrics"""
        avg_metric = (self.trust_level + self.respect_level + (self.philosophical_alignment + 1) / 2) / 3
        
        if avg_metric >= 0.8:
            self.relationship_status = RelationshipStatus.ALLIED
        elif avg_metric >= 0.6:
            self.relationship_status = RelationshipStatus.TRUSTING
        elif avg_metric >= 0.4:
            self.relationship_status = RelationshipStatus.RESPECTFUL
        elif avg_metric >= 0.2:
            self.relationship_status = RelationshipStatus.NEUTRAL
        elif avg_metric >= 0.1:
            self.relationship_status = RelationshipStatus.SUSPICIOUS
        else:
            self.relationship_status = RelationshipStatus.HOSTILE
    
    def get_conversation_context(self, topic: str) -> Dict:
        """Get relevant context for a new conversation about a topic"""
        context = {
            "relationship_status": self.relationship_status.value,
            "trust_level": self.trust_level,
            "respect_level": self.respect_level,
            "philosophical_alignment": self.philosophical_alignment,
            "previous_discussions": [],
            "known_player_beliefs": self.known_player_beliefs.copy(),
            "topics_to_avoid": list(self.topics_to_avoid),
            "topics_of_interest": list(self.topics_of_interest)
        }
        
        # Add topic-specific history
        if topic in self.topic_histories:
            topic_hist = self.topic_histories[topic]
            context["topic_history"] = {
                "times_discussed": topic_hist.times_discussed,
                "current_stance": topic_hist.current_stance.value,
                "previous_arguments": topic_hist.key_arguments_heard[-3:],  # Last 3 arguments
                "npc_responses": topic_hist.npc_counterarguments_used[-3:]  # Last 3 responses
            }
        
        # Add recent conversation memories
        recent_memories = [m for m in self.conversation_memories[-5:]]  # Last 5 conversations
        for memory in recent_memories:
            context["previous_discussions"].append({
                "topic": memory.topic_discussed,
                "player_stance": memory.player_stance,
                "npc_tone": memory.npc_response_tone.value,
                "impact": memory.relationship_impact,
                "memorable_quotes": memory.memorable_quotes
            })
        
        return context
    
    def apply_time_decay(self):
        """Apply relationship decay over time without interaction"""
        if self.last_interaction:
            days_since = (datetime.now() - self.last_interaction).days
            if days_since > 0:
                decay = self.relationship_decay_rate * days_since
                self.trust_level = max(0.0, self.trust_level - decay)
                self.respect_level = max(0.0, self.respect_level - decay * 0.5)  # Respect decays slower
                self.update_relationship_status()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "npc_id": self.npc_id,
            "npc_name": self.npc_name,
            "faction": self.faction,
            "relationship_status": self.relationship_status.value,
            "trust_level": self.trust_level,
            "respect_level": self.respect_level,
            "philosophical_alignment": self.philosophical_alignment,
            "known_player_beliefs": self.known_player_beliefs,
            "player_faction_alignment_perceived": self.player_faction_alignment_perceived,
            "topics_to_avoid": list(self.topics_to_avoid),
            "topics_of_interest": list(self.topics_of_interest),
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "conversation_count": len(self.conversation_memories)
        }

class RelationshipManager:
    """Manages all NPC relationships with the player"""
    
    def __init__(self):
        self.npc_profiles: Dict[str, NPCRelationshipProfile] = {}
        self.global_reputation: Dict[str, float] = {  # Faction-wide reputation
            "ashvattha": 0.0,
            "vaikuntha": 0.0, 
            "yuga_striders": 0.0,
            "shroud_mantra": 0.0
        }
    
    def create_npc_profile(self, npc_id: str, npc_name: str, faction: str) -> NPCRelationshipProfile:
        """Create a new NPC relationship profile"""
        profile = NPCRelationshipProfile(
            npc_id=npc_id,
            npc_name=npc_name,
            faction=faction,
            relationship_status=RelationshipStatus.NEUTRAL,
            trust_level=0.5,
            respect_level=0.5,
            philosophical_alignment=0.0
        )
        self.npc_profiles[npc_id] = profile
        return profile
    
    def get_npc_profile(self, npc_id: str) -> Optional[NPCRelationshipProfile]:
        """Get NPC relationship profile"""
        return self.npc_profiles.get(npc_id)
    
    def update_global_reputation(self, faction: str, change: float):
        """Update faction-wide reputation"""
        if faction in self.global_reputation:
            self.global_reputation[faction] = max(-1.0, min(1.0, 
                self.global_reputation[faction] + change))
    
    def get_faction_reputation(self, faction: str) -> float:
        """Get player's reputation with a faction"""
        return self.global_reputation.get(faction, 0.0)
    
    def save_relationships(self, filepath: str):
        """Save all relationships to file"""
        data = {
            "npc_profiles": {npc_id: profile.to_dict() 
                           for npc_id, profile in self.npc_profiles.items()},
            "global_reputation": self.global_reputation
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_relationships(self, filepath: str):
        """Load relationships from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.global_reputation = data.get("global_reputation", {})
            # Note: Full NPC profile loading would require reconstructing the dataclass
            # This is a simplified version for the basic structure
            
        except FileNotFoundError:
            print(f"No relationship file found at {filepath}, starting fresh")
        except Exception as e:
            print(f"Error loading relationships: {e}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_conversation_impact(player_stance: str, npc_faction: str, topic: str, 
                                hostility_level: str) -> float:
    """Calculate the relationship impact of a conversation"""
    # This would use the faction profiles and philosophical topics
    # to determine how much a conversation affects the relationship
    
    base_impact = 0.0
    
    # Positive impact for alignment with faction beliefs
    # Negative impact for contradiction
    # Modified by hostility level and topic sensitivity
    
    # This is a simplified version - full implementation would reference
    # the faction_profiles and philosophical_topics modules
    
    if hostility_level == "hostile":
        return -0.3  # Hostile topics always damage relationship
    elif hostility_level == "contemptuous":
        return -0.2
    elif hostility_level == "dismissive":
        return -0.1
    else:
        return 0.1  # Neutral topics slightly improve relationship
        
def get_appropriate_conversation_tone(relationship_status: RelationshipStatus, 
                                    faction: str, topic_hostility: bool) -> ConversationTone:
    """Determine appropriate conversation tone based on relationship and context"""
    
    if topic_hostility:
        if relationship_status in [RelationshipStatus.HOSTILE, RelationshipStatus.SUSPICIOUS]:
            return ConversationTone.AGGRESSIVE
        else:
            return ConversationTone.DISMISSIVE
    
    if relationship_status == RelationshipStatus.ALLIED:
        return ConversationTone.INTIMATE
    elif relationship_status == RelationshipStatus.TRUSTING:
        return ConversationTone.FRIENDLY
    elif relationship_status == RelationshipStatus.RESPECTFUL:
        return ConversationTone.PHILOSOPHICAL
    elif relationship_status == RelationshipStatus.NEUTRAL:
        return ConversationTone.FORMAL
    elif relationship_status == RelationshipStatus.SUSPICIOUS:
        return ConversationTone.DISMISSIVE
    else:  # HOSTILE
        return ConversationTone.AGGRESSIVE