# Angaraka Engine AI Integration Guide

## Model Files (Engine Format)
- `faction_dialogue_ashvattha.onnx` + `.meta` - Ashvattha model and metadata
- `faction_dialogue_vaikuntha.onnx` + `.meta` - Vaikuntha model and metadata  
- `faction_dialogue_yuga_striders.onnx` + `.meta` - Yuga Striders model and metadata
- `faction_dialogue_shroud_mantra.onnx` + `.meta` - Shroud Mantra model and metadata
- `tokenizer_vocab.json` - Vocabulary for tokenization
- `special_tokens.json` - Special token mappings

## Engine Integration
```cpp
// Load model using Angaraka.AI resource system
auto modelResource = std::make_shared<AIModelResource>("dialogue_ashvattha");
bool loaded = modelResource->Load("faction_dialogue_ashvattha.onnx");

// Metadata is automatically loaded from .meta file
const auto& metadata = modelResource->GetMetadata();
// metadata.inputNames[0] == "input"
// metadata.outputNames[0] == "output"
```

## Input Format for Engine
```cpp
// Tokenize player input (implement tokenizer in C++)
std::string prompt = "[ASHVATTHA] Player: What is karma? Assistant:";
std::vector<int64_t> input_ids = tokenize(prompt);  // Max 128 tokens

// Create ONNX tensor
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
    memory_info, input_ids.data(), input_ids.size(), 
    input_shape.data(), input_shape.size());

std::vector<Ort::Value> inputs;
inputs.push_back(std::move(input_tensor));

// Run inference using AIModelResource
auto outputs = modelResource->RunInference(inputs);
```

## Faction Prompt Format
- Ashvattha: `[ASHVATTHA] Player: {input} Assistant:`
- Vaikuntha: `[VAIKUNTHA] Player: {input} Assistant:`  
- Yuga Striders: `[YUGA_STRIDERS] Player: {input} Assistant:`
- Shroud Mantra: `[SHROUD_MANTRA] Player: {input} Assistant:`

## Output Processing
```cpp
// Extract logits from output tensor
float* logits = outputs[0].GetTensorMutableData<float>();
// Shape: [1, sequence_length, 50257]

// Implement sampling/generation logic in C++
// - Temperature sampling
// - Top-p filtering  
// - Token decoding using vocabulary
```

## Performance Targets (Validated)
- Inference time: <100ms (meets metadata.maxInferenceTimeMs)
- Memory usage: ~600MB per model (metadata.maxMemoryMB)
- Input length: Up to 128 tokens
- Output length: 80 new tokens recommended

## Integration with DialogueSystem
```cpp
DialogueRequest request;
request.factionId = "ashvattha";
request.playerInput = "What is karma?";

auto response = aiManager->GenerateDialogueSync(request);
// response.response contains the generated NPC dialogue
```
