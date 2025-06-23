"""
Angaraka AI Configuration
Configuration for Threads of Kaliyuga AI models
"""

# Model configurations
DIALOGUE_CONFIG = {
    "max_length": 512,
    "temperature": 0.8,
    "top_p": 0.9,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 3,
}

TERRAIN_CONFIG = {
    "image_size": 512,
    "channels": 3,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 100,
}

# Faction configurations
FACTIONS = {
    "ashvattha": {
        "name": "Ashvattha Collective",
        "ideology": "Ancient wisdom preservation",
        "speech_patterns": ["traditional", "reverent", "historical"],
        "philosophy": "dharma_traditional"
    },
    "vaikuntha": {
        "name": "Vaikuntha Initiative", 
        "ideology": "Karma quantification",
        "speech_patterns": ["precise", "analytical", "systematic"],
        "philosophy": "dharma_algorithmic"
    },
    "yuga_striders": {
        "name": "Yuga Striders",
        "ideology": "Revolutionary chaos",
        "speech_patterns": ["rebellious", "destructive", "anti-establishment"],
        "philosophy": "karma_rejection"
    }
}

# Hardware configuration
DEVICE_CONFIG = {
    "device": "cuda",
    "mixed_precision": True,
    "max_memory_gb": 20,  # Leave some VRAM for system
}

# Export configuration
EXPORT_CONFIG = {
    "onnx_opset": 17,
    "optimize_for_inference": True,
    "quantization": False,  # Disable for now, we have plenty of VRAM
}
