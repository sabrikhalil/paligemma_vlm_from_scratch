import torch
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
from processing_paligemma import PaliGemmaProcessor

# Load a tokenizer compatible with a general model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define a special token for images, assuming your model can handle this setup
image_token = "<image>"
tokens_to_add = {"additional_special_tokens": [image_token]}
tokenizer.add_special_tokens(tokens_to_add)

# Create an instance of your processor with the correct configurations
processor = PaliGemmaProcessor(tokenizer, image_size=224, num_image_tokens=10)

# Load a test image (this should be an actual image file on your disk)
image_path = './test_images/cat.jpg'
image = Image.open(image_path).convert('RGB')

# Test input text
test_text = ["This is a test prompt."]

# Execute the processor
outputs = processor(text=test_text, images=[image])

# Check the output types and shapes
print("Input IDs shape:", outputs['input_ids'].shape)
print("Attention Mask shape:", outputs['attention_mask'].shape)
print("Pixel Values shape:", outputs['pixel_values'].shape)
