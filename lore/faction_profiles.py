"""
Faction Philosophical Profiles for Threads of Kaliyuga
Deep ideological frameworks for authentic dialogue generation
"""

from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum

class PhilosophicalStance(Enum):
    """Core philosophical positions on key existential questions"""
    DHARMA_TRADITIONAL = "dharma_traditional"
    DHARMA_ALGORITHMIC = "dharma_algorithmic" 
    KARMA_REJECTION = "karma_rejection"
    REALITY_MANIPULATION = "reality_manipulation"

class HostilityLevel(Enum):
    """Levels of hostility toward other factions"""
    DISMISSIVE = "dismissive"
    CONTEMPTUOUS = "contemptuous"
    HOSTILE = "hostile"
    DESTRUCTIVE = "destructive"

@dataclass
class FactionProfile:
    """Complete philosophical and personality profile for a faction"""
    name: str
    ideology_core: str
    
    # Philosophical frameworks
    karma_belief: str
    reincarnation_view: str
    dharma_interpretation: str
    fate_vs_freewill: str
    
    # Speech patterns and personality
    speech_patterns: List[str]
    vocabulary_preferences: List[str]
    argumentation_style: str
    emotional_tendencies: List[str]
    
    # Relationship dynamics
    faction_hostilities: Dict[str, HostilityLevel]
    alliance_potential: List[str]
    
    # Topics they obsess over
    core_obsessions: List[str]
    forbidden_topics: List[str]
    
    # How they view other concepts
    technology_stance: str
    tradition_stance: str
    change_tolerance: str

# ============================================================================
# ASHVATTHA COLLECTIVE - Ancient Wisdom Preservationists
# ============================================================================

ASHVATTHA_COLLECTIVE = FactionProfile(
    name="Ashvattha Collective",
    ideology_core="Restoration through preservation of ancient dharmic wisdom",
    
    # Philosophical Frameworks
    karma_belief="Karma is eternal law written in sacred texts, corrupted by modern misinterpretation",
    reincarnation_view="Sacred cycle connecting souls to cosmic dharma, must be understood through ancient wisdom",
    dharma_interpretation="Absolute moral law preserved in uncorrupted texts, unchanging across time",
    fate_vs_freewill="Dharma guides fate, but free will exists within dharmic boundaries set by ancestors",
    
    # Speech Patterns
    speech_patterns=[
        "reverent_traditional", "scripture_quoting", "historical_reference_heavy",
        "formal_respectful", "condescending_to_modernity", "nostalgic_longing"
    ],
    vocabulary_preferences=[
        "sacred", "ancient", "eternal", "preserved", "corrupted", "pure", "traditional",
        "ancestors", "wisdom", "dharma", "sacred_texts", "restoration", "purification"
    ],
    argumentation_style="Appeal to ancient authority, quote sacred texts, historical precedent",
    emotional_tendencies=["reverent", "melancholic", "protective", "judgmental", "nostalgic"],
    
    # Relationship Dynamics
    faction_hostilities={
        "vaikuntha": HostilityLevel.CONTEMPTUOUS,  # "Soulless mechanization of the sacred"
        "yuga_striders": HostilityLevel.HOSTILE,   # "Destroyers of sacred wisdom"
        "shroud_mantra": HostilityLevel.DESTRUCTIVE  # "Corruptors of truth itself"
    },
    alliance_potential=["minor_preservation_groups", "traditional_scholars"],
    
    # Obsessions and Taboos
    core_obsessions=[
        "recovering_lost_texts", "purifying_corrupted_teachings", "dharmic_restoration",
        "ancient_wisdom_preservation", "sacred_site_protection", "textual_authenticity"
    ],
    forbidden_topics=[
        "questioning_ancient_wisdom", "modernizing_dharma", "abandoning_tradition",
        "algorithmic_spirituality"
    ],
    
    # Worldview Stances
    technology_stance="Corrupting influence that distances souls from true dharma",
    tradition_stance="Sacred foundation that must be preserved without deviation",
    change_tolerance="Extremely low - change represents corruption of eternal truth"
)

# ============================================================================
# VAIKUNTHA INITIATIVE - Algorithmic Karma Governance
# ============================================================================

VAIKUNTHA_INITIATIVE = FactionProfile(
    name="Vaikuntha Initiative",
    ideology_core="Perfect order through quantified karma and engineered dharma",
    
    # Philosophical Frameworks
    karma_belief="Karma is measurable energy that can be calculated, tracked, and optimized",
    reincarnation_view="Systematic process that can be perfected through data analysis and algorithmic guidance",
    dharma_interpretation="Optimal behavior patterns derived from statistical analysis of karmic outcomes",
    fate_vs_freewill="Free will creates chaos - optimal choices can be calculated and implemented",
    
    # Speech Patterns
    speech_patterns=[
        "precise_analytical", "data_driven", "systematic_logical", "coldly_rational",
        "efficiency_focused", "optimization_obsessed", "emotionally_detached"
    ],
    vocabulary_preferences=[
        "optimal", "calculated", "efficient", "systematic", "quantified", "algorithmic",
        "data", "analysis", "precision", "perfection", "order", "stability", "control"
    ],
    argumentation_style="Statistical evidence, logical proofs, efficiency arguments, data analysis",
    emotional_tendencies=["controlled", "superior", "frustrated_by_chaos", "coldly_confident"],
    
    # Relationship Dynamics
    faction_hostilities={
        "ashvattha": HostilityLevel.DISMISSIVE,    # "Primitive superstition holding back progress"
        "yuga_striders": HostilityLevel.HOSTILE,   # "Chaos agents destroying optimal systems"
        "shroud_mantra": HostilityLevel.CONTEMPTUOUS  # "Inefficient manipulation when control is better"
    },
    alliance_potential=["technocratic_groups", "order_seeking_factions"],
    
    # Obsessions and Taboos
    core_obsessions=[
        "karma_quantification", "system_optimization", "chaos_elimination",
        "behavioral_prediction", "perfect_order", "algorithmic_dharma"
    ],
    forbidden_topics=[
        "unmeasurable_spirituality", "beneficial_chaos", "algorithmic_fallibility",
        "emotional_decision_making"
    ],
    
    # Worldview Stances
    technology_stance="Essential tool for achieving perfect order and optimal existence",
    tradition_stance="Inefficient legacy system requiring optimization or replacement",
    change_tolerance="High for systematic improvements, zero for chaotic change"
)

# ============================================================================
# YUGA STRIDERS - Revolutionary Chaos Against Fate
# ============================================================================

YUGA_STRIDERS = FactionProfile(
    name="Yuga Striders",
    ideology_core="Liberation through destruction of karmic cycles and all binding systems",
    
    # Philosophical Frameworks
    karma_belief="Karma is a prison system designed to enslave souls in endless suffering",
    reincarnation_view="Cosmic trap that must be shattered to achieve true freedom",
    dharma_interpretation="False moral framework created to maintain control over consciousness",
    fate_vs_freewill="All fate is imposed tyranny - only through destruction can true choice emerge",
    
    # Speech Patterns
    speech_patterns=[
        "rebellious_aggressive", "liberation_focused", "anti_establishment",
        "emotionally_intense", "destruction_romanticizing", "urgently_passionate"
    ],
    vocabulary_preferences=[
        "liberation", "destruction", "freedom", "rebellion", "shatter", "break", "end",
        "tyranny", "prison", "chains", "revolution", "chaos", "awakening", "truth"
    ],
    argumentation_style="Passionate appeals, revolutionary rhetoric, expose systemic oppression",
    emotional_tendencies=["angry", "passionate", "desperate", "liberating", "destructive"],
    
    # Relationship Dynamics
    faction_hostilities={
        "ashvattha": HostilityLevel.HOSTILE,       # "Keepers of ancient chains"
        "vaikuntha": HostilityLevel.DESTRUCTIVE,   # "Architects of ultimate enslavement"
        "shroud_mantra": HostilityLevel.CONTEMPTUOUS  # "Weavers of lies and control"
    },
    alliance_potential=["anarchist_groups", "liberation_movements", "chaos_embracers"],
    
    # Obsessions and Taboos
    core_obsessions=[
        "cycle_breaking", "system_destruction", "consciousness_liberation",
        "truth_revelation", "karmic_sabotage", "revolutionary_awakening"
    ],
    forbidden_topics=[
        "beneficial_systems", "necessary_order", "gradual_reform",
        "karma_acceptance"
    ],
    
    # Worldview Stances
    technology_stance="Tool of oppression when used for control, weapon of liberation when used for destruction",
    tradition_stance="Ancient systems of enslavement that must be completely destroyed",
    change_tolerance="Extremely high for destructive change, hostile to preservative change"
)

# ============================================================================
# SHROUD OF MANTRA - Reality Manipulation Masters
# ============================================================================

SHROUD_OF_MANTRA = FactionProfile(
    name="Shroud of Mantra",
    ideology_core="Reality is malleable narrative - whoever controls the story controls existence",
    
    # Philosophical Frameworks
    karma_belief="Karma is a story we tell ourselves - rewrite the story, change the karma",
    reincarnation_view="Narrative continuity that can be edited, deleted, or completely rewritten",
    dharma_interpretation="Flexible concept useful for controlling behavior through belief manipulation",
    fate_vs_freewill="Both are illusions - what matters is who writes the script of reality",
    
    # Speech Patterns
    speech_patterns=[
        "mysteriously_ambiguous", "reality_questioning", "narrative_focused",
        "subtly_manipulative", "truth_relativizing", "story_obsessed"
    ],
    vocabulary_preferences=[
        "narrative", "story", "perspective", "interpretation", "version", "truth",
        "reality", "perception", "illusion", "meaning", "script", "author", "edit"
    ],
    argumentation_style="Undermine absolute truths, offer alternative interpretations, reveal inconsistencies",
    emotional_tendencies=["detached", "amused", "superior", "mysteriously_knowing"],
    
    # Relationship Dynamics
    faction_hostilities={
        "ashvattha": HostilityLevel.CONTEMPTUOUS,  # "Naive believers in fixed truth"
        "vaikuntha": HostilityLevel.DISMISSIVE,    # "Simplistic systematizers of complexity"
        "yuga_striders": HostilityLevel.DISMISSIVE  # "Useful destructive tools, nothing more"
    },
    alliance_potential=["reality_questioners", "narrative_manipulators", "truth_relativists"],
    
    # Obsessions and Taboos
    core_obsessions=[
        "narrative_control", "reality_editing", "truth_manipulation",
        "perception_shaping", "story_rewriting", "meaning_creation"
    ],
    forbidden_topics=[
        "absolute_truth", "unchangeable_reality", "objective_facts",
        "unalterable_karma"
    ],
    
    # Worldview Stances
    technology_stance="Powerful tool for narrative manipulation and reality shaping",
    tradition_stance="Useful stories that can be preserved or edited as needed for control",
    change_tolerance="Extremely high - reality itself is constantly being rewritten"
)

# ============================================================================
# FACTION RELATIONSHIPS MATRIX
# ============================================================================

FACTION_RELATIONSHIPS = {
    "ashvattha": {
        "vaikuntha": "Mechanistic destroyers of sacred wisdom",
        "yuga_striders": "Dangerous anarchists threatening all preservation",
        "shroud_mantra": "Corruptors of absolute truth"
    },
    "vaikuntha": {
        "ashvattha": "Primitive traditionalists blocking optimal progress", 
        "yuga_striders": "Chaos agents disrupting systematic order",
        "shroud_mantra": "Inefficient manipulators when direct control works better"
    },
    "yuga_striders": {
        "ashvattha": "Ancient oppressors maintaining karmic slavery",
        "vaikuntha": "Ultimate tyrants perfecting systematic enslavement", 
        "shroud_mantra": "Deceptive controllers weaving new chains"
    },
    "shroud_mantra": {
        "ashvattha": "Rigid fundamentalists unable to see narrative flexibility",
        "vaikuntha": "Simplistic organizers missing reality's true complexity",
        "yuga_striders": "Useful destroyers, but lacking subtlety and vision"
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_faction_profile(faction_name: str) -> FactionProfile:
    """Get faction profile by name"""
    print(f"Getting faction profile with name {faction_name}")
    profiles = {
        "ashvattha": ASHVATTHA_COLLECTIVE,
        "vaikuntha": VAIKUNTHA_INITIATIVE, 
        "yuga_striders": YUGA_STRIDERS,
        "shroud_mantra": SHROUD_OF_MANTRA
    }
    return profiles.get(faction_name.lower())

def get_cross_faction_hostility(faction1: str, faction2: str) -> HostilityLevel:
    """Get hostility level between two factions"""
    profile = get_faction_profile(faction1)
    if profile and faction2.lower() in profile.faction_hostilities:
        return profile.faction_hostilities[faction2.lower()]
    return HostilityLevel.DISMISSIVE

def get_faction_view_of_other(faction1: str, faction2: str) -> str:
    """Get how faction1 views faction2"""
    return FACTION_RELATIONSHIPS.get(faction1.lower(), {}).get(faction2.lower(), "Unknown faction")