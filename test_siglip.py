import torch
import time
from PIL import Image
from torchvision import transforms
from modeling_siglip import SiglipVisionConfig, SiglipVisionEmbeddings, SiglipVisionEncoder

def test_vision_embeddings():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the image
    image_path = './test_images/cat.jpg'
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations: resize, convert to tensor, and move to GPU
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device))
    ])
    
    # Apply the transformations to the image (including moving to GPU)
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension [1, C, H, W]
    # Initialize the config and VisionEmbeddings class directly with CUDA
    config = SiglipVisionConfig()
    vision_embeddings = SiglipVisionEmbeddings(config).to(device)
    
    # Ensure the model is in evaluation mode
    vision_embeddings.eval()
    
    # Use torch.no_grad() to disable gradient calculations
    with torch.no_grad():
        # Get the embeddings
        embeddings = vision_embeddings(pixel_values)
    
    # Print the outputs
    print(f"Embeddings size: {embeddings.size()}")
    print(f"Total time: {time.time() - start_time:.4f} seconds")




def test_vision_encoder():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the image
    image_path = './test_images/cat.jpg'
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations: resize, convert to tensor, and move to GPU
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to fit the model's expected input size
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Lambda(lambda x: x.to(device))  # Move the tensor to the GPU
    ])
    
    # Apply the transformations to the image (including moving to GPU)
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension [1, C, H, W]
    
    # Initialize the config and the relevant classes directly on the GPU
    config = SiglipVisionConfig()
    vision_embeddings = SiglipVisionEmbeddings(config).to(device)
    vision_encoder = SiglipVisionEncoder(config).to(device)
    
    # Ensure the models are in evaluation mode
    vision_embeddings.eval()
    vision_encoder.eval()
    
    # Use torch.no_grad() to disable gradient calculations
    with torch.no_grad():
        # Get the embeddings
        embeddings = vision_embeddings(pixel_values)
        # Encode the embeddings
        print("embeddings shape", embeddings.shape)
        encoded_output = vision_encoder(embeddings)
    
    # Print the outputs
    print(f"Embeddings size: {embeddings.size()}")
    print(f"Encoded output size: {encoded_output.size()}")
    print(f"Total time: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    test_vision_encoder()
    # test_vision_embeddings()