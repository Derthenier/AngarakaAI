"""
Python Import Setup for Angaraka AI
Fixes module import paths for local development
"""

import sys
import os

# Add the current directory to Python path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from the script location
sys.path.insert(0, project_root)

print(f"Added to Python path: {project_root}")
print("Python import paths configured!")

# Test imports to verify everything works
try:
    from lore.faction_profiles import get_faction_profile, FACTION_RELATIONSHIPS
    from lore.philosophical_topics import get_topic_by_name, TopicComplexity
    from models.dialogue.relationship_tracker import RelationshipStatus, ConversationTone
    print("✅ All imports successful!")
    print("✅ Ready to generate dialogue datasets!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all the artifact files are saved in the correct locations.")