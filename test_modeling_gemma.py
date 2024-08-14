# Test Code Adjustments
import torch
from PIL import Image
from torchvision import transforms
from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
 
# Text configuration dictionary setup
text_config_params = {
    'vocab_size': 50265,
    'hidden_size': 2048,
    'intermediate_size': 5120,
    'num_hidden_layers': 12,
    'num_attention_heads': 16,
    'num_key_value_heads': 16,
#    'pad_token_id': 0,  # Include pad_token_id here
    'max_position_embeddings': 512
}
 
# Vision configuration dictionary setup
vision_config_params = {
    'image_size': 224,
    'patch_size': 16,
    'num_channels': 3,
    'embed_dim': 2048,
    'num_hidden_layers': 12,
    'num_attention_heads': 12
}
 
# Create a PaliGemma configuration using dictionaries directly
config = PaliGemmaConfig(
    vision_config=vision_config_params,
    text_config=text_config_params,
    image_token_index=50264  # Assuming 50264 is used as the special token index for images
)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device)
 
# Initialize the model with the configuration instance
model = PaliGemmaForConditionalGeneration(config).to(device)
model.eval()  # Set the model to evaluation mode
 
# Create dummy inputs
input_ids = torch.randint(0, 50265, (1, 128)).to(device)  # Random input_ids assuming vocab size
attention_mask = torch.ones(1, 128).to(device)  # Simulate full attention
pixel_values = torch.rand(1, 3, 224, 224).to(device)  # Random pixel values for an image
 
# Forward pass
outputs = model(input_ids, pixel_values, attention_mask)
 
# If the model returns a dictionary, print keys to understand its structure
print("Output keys:", outputs.keys())
 
# Assuming the output logits are stored under the key 'logits'
if 'logits' in outputs:
    print("Output logits shape:", outputs['logits'].shape)
else:
    print("Expected 'logits' key not found in model output")