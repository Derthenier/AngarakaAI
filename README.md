# AngarakaAI - Threads of Kaliyuga AI Models

> **Testing AI model generation for philosophical dialogue systems**

AngarakaAI is an experimental AI model generation project focused on creating sophisticated philosophical dialogue systems. The project explores faction-based AI personalities with distinct ideological frameworks, speech patterns, and philosophical stances, particularly within a "Threads of Kaliyuga" narrative universe.

## 🌟 Features

### Faction-Based AI Personalities
- **Four Distinct Factions** with unique philosophical frameworks:
  - **Ashvattha Collective**: Ancient wisdom preservationists with reverent, traditional speech patterns
  - **Vaikuntha Initiative**: Algorithmic karma governance advocates with precise, analytical communication
  - **Yuga Striders**: Revolutionary chaos agents with rebellious, liberation-focused rhetoric
  - **Shroud of Mantra**: Reality manipulation masters with mysteriously ambiguous discourse

### Advanced Dialogue Generation
- **Faction-Specific Speech Patterns**: Each faction has distinct vocabulary, argumentation styles, and emotional tendencies
- **Cross-Faction Dynamics**: Complex hostility matrices and relationship systems between factions
- **Philosophical Depth**: Deep ideological frameworks covering karma, dharma, reincarnation, and consciousness
- **Data Cleaning & Enhancement**: Sophisticated dialogue data processing for coherent faction responses

### AI Model Training Pipeline
- **Dialogue Model Training**: Fine-tuning systems for faction-specific conversation patterns
- **Terrain Generation**: AI models for environmental/world generation
- **Synthetic Data Generation**: Automated creation of training data for philosophical discussions
- **Model Export & Compression**: ONNX export with optimization for inference

## 📁 Project Structure

```
AngarakaAI/
├── config/                    # Configuration files
│   ├── __init__.py
│   └── config.py              # Model configs, faction settings, hardware specs
├── data/                      # Training and validation datasets
├── export/                    # Model export and compression utilities
├── lore/                      # Philosophical framework definitions
│   ├── faction_profiles.py    # Complete faction personality profiles
│   └── philosophical_topics.py # Core philosophical discussion topics
├── models/                    # Trained AI models and architectures
├── scripts/                   # Utility scripts and automation
├── synthetic_data_generators/ # Data generation for training
├── tests/                     # Test suites and validation
├── training/                  # Model training implementations
│   ├── dialogue/              # Dialogue model training
│   └── terrain/               # Terrain generation training
├── utils/                     # Shared utilities and helpers
├── validation/                # Model validation and testing
├── clean_dialogue_data.py     # Data cleaning and enhancement tool
└── setup_imports.py           # Project import configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- PyTorch and transformers library
- 20GB+ available VRAM (configured in hardware settings)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Derthenier/AngarakaAI.git
   cd AngarakaAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # Create this file based on imports
   ```

3. **Set up imports**:
   ```bash
   python setup_imports.py
   ```

### Basic Usage

#### 1. Clean and Enhance Dialogue Data
```bash
python clean_dialogue_data.py
```
This will:
- Clean existing conversation data in `data/synthetic/dialogue/`
- Generate enhanced training examples
- Output cleaned data to `data/synthetic/dialogue_cleaned/`
- Create enhanced examples in `data/synthetic/dialogue_enhanced/`

#### 2. Explore Faction Profiles
```python
from lore.faction_profiles import get_faction_profile

# Get a faction's complete philosophical framework
ashvattha = get_faction_profile("ashvattha")
print(f"Karma belief: {ashvattha.karma_belief}")
print(f"Speech patterns: {ashvattha.speech_patterns}")
```

#### 3. Train Dialogue Models
```bash
# Navigate to training directory and run dialogue training
cd training/dialogue
python train_dialogue_model.py
```

## 🎯 Core Components

### Faction System
The heart of AngarakaAI is its faction system, where each faction represents a distinct philosophical worldview:

- **Philosophical Frameworks**: Each faction has unique interpretations of karma, dharma, reincarnation, and consciousness
- **Speech Pattern Recognition**: Faction-specific vocabulary, argumentation styles, and emotional tendencies
- **Relationship Dynamics**: Complex hostility matrices between factions (from dismissive to destructive)
- **Topic Obsessions**: Each faction has core topics they focus on and forbidden subjects they avoid

### Dialogue Data Processing
The `clean_dialogue_data.py` system provides:
- **Quality Enhancement**: Removes problematic patterns and improves coherence
- **Faction Validation**: Ensures responses match faction philosophical stances
- **Vocabulary Enhancement**: Adds faction-appropriate terminology and speech patterns
- **Template Generation**: Creates high-quality training examples for each faction

### Configuration Management
Centralized configuration in `config/config.py` covers:
- **Model Parameters**: Learning rates, batch sizes, sequence lengths
- **Hardware Settings**: CUDA configuration, memory limits, mixed precision
- **Export Options**: ONNX optimization, quantization settings
- **Faction Definitions**: Core philosophical stances and relationships

## 🔧 Configuration

### Model Configuration
```python
DIALOGUE_CONFIG = {
    "max_length": 512,
    "temperature": 0.8,
    "top_p": 0.9,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 3,
}
```

### Hardware Configuration
```python
DEVICE_CONFIG = {
    "device": "cuda",
    "mixed_precision": True,
    "max_memory_gb": 20,  # Leave some VRAM for system
}
```

### Faction Customization
Factions can be customized by modifying their profiles in `lore/faction_profiles.py`. Each faction includes:
- Core ideology and philosophical stances
- Speech patterns and vocabulary preferences
- Relationship dynamics with other factions
- Topics of obsession and forbidden subjects

## 🧪 Testing & Validation

The project includes comprehensive testing systems:

- **Unit Tests**: Located in `tests/` directory
- **Dialogue Quality Validation**: Ensures faction responses match philosophical frameworks
- **Model Performance Testing**: Validates training convergence and output quality
- **Cross-Faction Consistency**: Tests relationship dynamics and hostility systems

## 📊 Data Management

### Training Data Structure
```json
{
  "conversations": [
    {
      "id": "conversation_001",
      "npc_faction": "ashvattha",
      "main_topic": "karma",
      "dialogue": [
        {"speaker": "player", "text": "What is karma?"},
        {"speaker": "npc", "text": "The ancient texts tell us that karma is eternal law..."}
      ]
    }
  ]
}
```

### Faction Data Organization
- Raw dialogue data: `data/synthetic/dialogue/`
- Cleaned dialogue data: `data/synthetic/dialogue_cleaned/`
- Enhanced examples: `data/synthetic/dialogue_enhanced/`
- Model checkpoints: `models/`
- Export outputs: `export/`

## 🚀 Advanced Usage

### Custom Faction Creation
1. Define faction profile in `lore/faction_profiles.py`
2. Add faction relationships to existing factions
3. Create training data with new faction responses
4. Train models with updated faction set

### Model Export & Deployment
```python
# Export trained model to ONNX format
from export import export_model
export_model(
    model_path="models/dialogue_model.pt",
    output_path="export/dialogue_model.onnx",
    optimize=True
)
```

### Synthetic Data Generation
Generate additional training data:
```bash
cd synthetic_data_generators
python generate_faction_dialogues.py --faction ashvattha --count 1000
```

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies
4. Make your changes
5. Run tests: `python -m pytest tests/`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Contribution Guidelines
- **Faction Consistency**: Ensure new content matches established philosophical frameworks
- **Code Quality**: Follow Python PEP 8 standards
- **Documentation**: Update relevant documentation for new features
- **Testing**: Add tests for new functionality
- **Philosophical Accuracy**: Maintain depth and consistency in faction worldviews

### Areas for Contribution
- **New Philosophical Topics**: Expand the range of discussable subjects
- **Faction Relationships**: Develop more nuanced inter-faction dynamics
- **Training Optimization**: Improve model training efficiency and quality
- **Data Quality**: Enhance dialogue cleaning and generation systems
- **Model Architecture**: Experiment with new AI model designs

## 📋 TODO & Roadmap

### Current Development
- [ ] Implement terrain generation models
- [ ] Expand philosophical topic coverage
- [ ] Enhance cross-faction dialogue systems
- [ ] Optimize model export pipeline
- [ ] Add real-time dialogue inference

### Future Plans
- [ ] Multi-modal personality expression (text + behavior)
- [ ] Dynamic faction evolution based on interactions
- [ ] Integration with game engines
- [ ] Voice synthesis for faction-specific speech
- [ ] Advanced emotional modeling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by philosophical traditions and AI model research
- Built with PyTorch and the Transformers library
- Community contributions to philosophical AI development
- Research in personality-driven dialogue systems

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/Derthenier/AngarakaAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Derthenier/AngarakaAI/discussions)
- **Documentation**: Check the `docs/` folder for detailed guides

---

*"In the threads of Kaliyuga, every voice carries the weight of cosmic philosophy, and every dialogue shapes the fabric of digital consciousness."*

**Project Status**: 🚧 Active Development  
**Last Updated**: September 2025  
**Language**: Python 100%
