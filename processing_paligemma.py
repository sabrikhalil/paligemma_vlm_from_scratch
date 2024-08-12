import numpy as np 
from PIL import Image
import torch    
from transformers import PreTrainedTokenizer
import torchvision.transforms as transforms


IMAGE_TOKEN = "<image>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]


def build_string_from_input(prompt, bos_token, image_seq_len, image_token):
        """
        Builds a string from the input prompt and image tokens.
        For example, for the call:
        build_string_from_input(
            prompt="Prefix str"
            bos_token="<s>",
            image_seq_len=3,
            image_token="<im>",
        )
        The output will be:
        "<im><im><im><s>Initial str"
        Args:
            prompt (`List[Union[str, ImageInput]]`): The input prompt.
            bos_token (`str`): The beginning of sentence token.
            image_seq_len (`int`): The length of the image sequence.
            image_token (`str`): The image token.
        """
        return f"{image_token * image_seq_len}{bos_token}{prompt}\n"



class PaliGemmaProcessor:

    def __init__(self, tokenizer: PreTrainedTokenizer, image_size: int, num_image_tokens: int):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_seq_length = num_image_tokens
        self.image_token = IMAGE_TOKEN

        tokens_to_add = {"additional_special_tokens": [IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def __call__(self, text: list[str], images: list[Image.Image], padding="longest", truncation=True):
        assert len(images) == 1 and len(text) == 1, "The processor currently supports a single image and single text input only."

        input_strings = [
            build_string_from_input(
                prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=IMAGE_TOKEN,
            )
            for prompt in text
        ]

        print("input strings: ", input_strings)
        
        # Process the image
        pixel_values = self.process_image(images[0], self.image_size)

        print("pixel values : ", pixel_values)
        
        # Tokenize text input
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding=padding, truncation=truncation)
        print("tokenized inputs: ", inputs)
        
        # Prepare outputs
        outputs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'pixel_values': pixel_values
        }
        print("outputs : ", outputs)
        return outputs

    def process_image(self, image: Image.Image, size: int):
        # Resize and convert the image
        transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

    def add_image_tokens_to_prompt(self, prompt: str):
        # Prepend the specified number of image tokens to the prompt
        image_tokens = ' '.join([self.image_token for _ in range(self.image_seq_length)])
        return f"{image_tokens} {prompt}"
