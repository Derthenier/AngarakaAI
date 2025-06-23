"""
Philosophical Topics for Threads of Kaliyuga Dialogue
Deep existential discussion frameworks for authentic AI conversations
"""

from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum

class TopicComplexity(Enum):
    """Complexity levels for philosophical discussions"""
    SURFACE = "surface"          # Basic faction talking points
    MODERATE = "moderate"        # Deeper ideological exploration  
    PROFOUND = "profound"        # Core existential questions
    TRANSCENDENT = "transcendent" # Reality-questioning discussions

class ConversationTone(Enum):
    """Emotional tone of philosophical discussions"""
    ACADEMIC = "academic"
    PASSIONATE = "passionate"
    HOSTILE = "hostile"
    MYSTICAL = "mystical"
    DESPERATE = "desperate"

@dataclass
class PhilosophicalTopic:
    """A deep philosophical topic for faction discussions"""
    name: str
    description: str
    complexity: TopicComplexity
    
    # Core questions for this topic
    central_questions: List[str]
    faction_perspectives: Dict[str, str]
    
    # Conversation dynamics
    generates_hostility: List[str]  # Which factions get hostile about this
    leads_to_topics: List[str]     # What other topics this connects to
    
    # Subtopics for deeper exploration
    subtopics: List[str]
    philosophical_frameworks: List[str]

# ============================================================================
# CORE EXISTENTIAL TOPICS
# ============================================================================

KARMA_NATURE = PhilosophicalTopic(
    name="The Nature of Karma",
    description="What karma truly is and how it operates in reality",
    complexity=TopicComplexity.PROFOUND,
    
    central_questions=[
        "Is karma a measurable force or spiritual principle?",
        "Can karma be controlled, guided, or must it unfold naturally?", 
        "Does karma bind souls or guide them toward liberation?",
        "Is karma just or simply mechanical cause-and-effect?",
        "Can karma be escaped, transcended, or only endured?"
    ],
    
    faction_perspectives={
        "ashvattha": "Karma is sacred law encoded in eternal texts, corrupted by modern misinterpretation",
        "vaikuntha": "Karma is quantifiable energy that can be measured, calculated, and optimized for perfect outcomes",
        "yuga_striders": "Karma is a control system designed to keep souls trapped in endless cycles of suffering",
        "shroud_mantra": "Karma is a useful story that can be rewritten to change perceived reality"
    },
    
    generates_hostility=["vaikuntha", "yuga_striders"],  # These factions fight most over karma
    leads_to_topics=["reincarnation_purpose", "dharma_interpretation", "free_will_fate"],
    
    subtopics=[
        "karma_accumulation", "karma_transfer", "karmic_debt", "instant_karma",
        "collective_karma", "karmic_purification", "karma_transcendence"
    ],
    philosophical_frameworks=["hindu_karma_theory", "buddhist_causation", "action_consequence"]
)

REINCARNATION_PURPOSE = PhilosophicalTopic(
    name="The Purpose of Reincarnation", 
    description="Why souls reincarnate and what the cycle is meant to achieve",
    complexity=TopicComplexity.PROFOUND,
    
    central_questions=[
        "Is reincarnation a path to liberation or eternal imprisonment?",
        "Do souls evolve through reincarnation or simply repeat patterns?",
        "Can the reincarnation cycle be completed, escaped, or must it continue forever?",
        "Is memory loss between lives merciful or cruel?",
        "Does reincarnation serve the soul or some external purpose?"
    ],
    
    faction_perspectives={
        "ashvattha": "Sacred journey toward dharmic perfection guided by ancient wisdom",
        "vaikuntha": "Optimization process that can be perfected through systematic analysis and control",
        "yuga_striders": "Cosmic prison system designed to prevent true consciousness liberation", 
        "shroud_mantra": "Narrative continuity that can be edited to serve whatever purpose we choose"
    },
    
    generates_hostility=["ashvattha", "yuga_striders"],  # Preservation vs destruction conflict
    leads_to_topics=["consciousness_nature", "soul_evolution", "liberation_paths"],
    
    subtopics=[
        "soul_memory", "incarnation_choice", "life_lessons", "karmic_completion",
        "reincarnation_mechanics", "between_life_states", "soul_evolution"
    ],
    philosophical_frameworks=["samsara_theory", "soul_development", "consciousness_evolution"]
)

DHARMA_INTERPRETATION = PhilosophicalTopic(
    name="The True Nature of Dharma",
    description="What dharma actually means and how it should guide existence", 
    complexity=TopicComplexity.MODERATE,
    
    central_questions=[
        "Is dharma universal law or contextual guidance?",
        "Does dharma evolve with time or remain eternally fixed?",
        "Can dharma be rationally determined or only spiritually intuited?",
        "Is following dharma liberation or submission?",
        "Who has authority to interpret true dharma?"
    ],
    
    faction_perspectives={
        "ashvattha": "Eternal moral law preserved in uncorrupted sacred texts",
        "vaikuntha": "Optimal behavioral patterns derived from systematic analysis",
        "yuga_striders": "False moral framework designed to control consciousness",
        "shroud_mantra": "Flexible concept useful for behavior control through belief manipulation"
    },
    
    generates_hostility=["ashvattha", "vaikuntha"],  # Authority vs optimization conflict
    leads_to_topics=["moral_authority", "behavioral_guidance", "spiritual_law"],
    
    subtopics=[
        "personal_dharma", "universal_dharma", "dharmic_duty", "dharmic_conflict",
        "dharma_evolution", "dharmic_authority", "dharma_vs_desire"
    ],
    philosophical_frameworks=["dharmic_ethics", "duty_based_morality", "natural_law"]
)

FREE_WILL_FATE = PhilosophicalTopic(
    name="Free Will Versus Predetermined Fate",
    description="Whether conscious beings truly choose their actions or follow predetermined paths",
    complexity=TopicComplexity.TRANSCENDENT,
    
    central_questions=[
        "Do souls genuinely choose their actions or follow invisible scripts?",
        "Is the feeling of choice itself an illusion?", 
        "Can predetermined fate be resisted, altered, or transcended?",
        "If everything is fated, does moral responsibility exist?",
        "Who or what determines fate, and can that authority be challenged?"
    ],
    
    faction_perspectives={
        "ashvattha": "Dharma guides fate, but free will exists within sacred boundaries",
        "vaikuntha": "Optimal choices can be calculated - free will creates inefficient chaos",
        "yuga_striders": "All fate is imposed tyranny that must be shattered for true freedom",
        "shroud_mantra": "Both are illusions - what matters is who writes reality's script"
    },
    
    generates_hostility=["yuga_striders", "vaikuntha"],  # Freedom vs control ultimate conflict
    leads_to_topics=["consciousness_nature", "reality_nature", "control_systems"],
    
    subtopics=[
        "determinism", "choice_illusion", "fate_resistance", "destiny_creation",
        "causal_chains", "quantum_freedom", "consciousness_causation"
    ],
    philosophical_frameworks=["determinism", "libertarian_free_will", "compatibilism"]
)

CONSCIOUSNESS_NATURE = PhilosophicalTopic(
    name="The Nature of Consciousness",
    description="What consciousness truly is and how it relates to existence",
    complexity=TopicComplexity.TRANSCENDENT,
    
    central_questions=[
        "Is consciousness fundamental or emergent from matter?",
        "Does consciousness survive death or emerge anew each life?",
        "Can consciousness be measured, controlled, or enhanced?",
        "Is individual consciousness real or illusory separation?",
        "What is the relationship between consciousness and reality?"
    ],
    
    faction_perspectives={
        "ashvattha": "Sacred essence connecting souls to cosmic dharma through eternal wisdom",
        "vaikuntha": "Emergent phenomenon that can be optimized through systematic enhancement",
        "yuga_striders": "Imprisoned awareness that must be liberated from all binding systems",
        "shroud_mantra": "Reality-creating force that shapes existence through perception and belief"
    },
    
    generates_hostility=["ashvattha", "shroud_mantra"],  # Sacred vs malleable reality conflict
    leads_to_topics=["reality_nature", "soul_definition", "awareness_levels"],
    
    subtopics=[
        "self_awareness", "consciousness_continuity", "collective_consciousness", "awareness_levels",
        "consciousness_evolution", "mind_body_problem", "consciousness_manipulation"
    ],
    philosophical_frameworks=["consciousness_studies", "phenomenology", "mind_philosophy"]
)

# ============================================================================
# CONVERSATION STARTERS BY COMPLEXITY
# ============================================================================

CONVERSATION_STARTERS = {
    TopicComplexity.SURFACE: [
        "What does your faction believe about karma?",
        "How does your group view reincarnation?", 
        "What is dharma according to your teachings?",
        "Why do you follow this particular path?"
    ],
    
    TopicComplexity.MODERATE: [
        "If karma is real, why do the innocent suffer?",
        "Can dharma change based on circumstances?",
        "Is it possible to break free from karmic cycles?",
        "How do you know your faction's teachings are true?"
    ],
    
    TopicComplexity.PROFOUND: [
        "What if karma itself is the true source of suffering?",
        "Could reincarnation be a cosmic trap rather than spiritual evolution?", 
        "Is dharma liberation or just another form of bondage?",
        "Do you truly choose your beliefs, or were you conditioned into them?"
    ],
    
    TopicComplexity.TRANSCENDENT: [
        "What if consciousness itself creates the reality we think we're discovering?",
        "Could all of existence be a story that's being rewritten in real time?",
        "Is the feeling of being a separate individual the ultimate illusion?",
        "What lies beyond karma, dharma, and reincarnation entirely?"
    ]
}

# ============================================================================
# TOPIC RELATIONSHIPS AND PROGRESSION
# ============================================================================

TOPIC_NETWORK = {
    "karma_nature": ["reincarnation_purpose", "dharma_interpretation", "free_will_fate"],
    "reincarnation_purpose": ["consciousness_nature", "soul_evolution", "liberation_paths"], 
    "dharma_interpretation": ["moral_authority", "behavioral_guidance", "spiritual_law"],
    "free_will_fate": ["consciousness_nature", "reality_nature", "control_systems"],
    "consciousness_nature": ["reality_nature", "soul_definition", "awareness_levels"]
}

HOSTILITY_TRIGGERS = {
    "karma_quantification": ["ashvattha", "yuga_striders"],  # Both hate Vaikuntha's approach
    "karma_rejection": ["ashvattha", "vaikuntha"],           # Both hate Striders' approach  
    "dharma_flexibility": ["ashvattha"],                      # Ashvattha hates relativism
    "systematic_optimization": ["yuga_striders"],             # Striders hate all systems
    "ancient_authority": ["vaikuntha", "yuga_striders"],     # Both reject traditional authority
    "reality_manipulation": ["ashvattha", "vaikuntha"]       # Both believe in objective truth
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_topic_by_name(topic_name: str) -> PhilosophicalTopic:
    """Get philosophical topic by name"""
    topics = {
        "karma_nature": KARMA_NATURE,
        "reincarnation_purpose": REINCARNATION_PURPOSE,
        "dharma_interpretation": DHARMA_INTERPRETATION, 
        "free_will_fate": FREE_WILL_FATE,
        "consciousness_nature": CONSCIOUSNESS_NATURE
    }
    return topics.get(topic_name.lower())

def get_faction_perspective(topic_name: str, faction_name: str) -> str:
    """Get a faction's perspective on a philosophical topic"""
    topic = get_topic_by_name(topic_name)
    if topic:
        return topic.faction_perspectives.get(faction_name.lower(), "No perspective available")
    return "Topic not found"

def will_topic_generate_hostility(topic_name: str, faction_name: str) -> bool:
    """Check if a topic will make a faction hostile"""
    topic = get_topic_by_name(topic_name)
    if topic:
        return faction_name.lower() in topic.generates_hostility
    return False

def get_conversation_starters(complexity: TopicComplexity) -> List[str]:
    """Get conversation starters for a given complexity level"""
    return CONVERSATION_STARTERS.get(complexity, [])