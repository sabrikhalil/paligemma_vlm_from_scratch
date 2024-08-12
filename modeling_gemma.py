from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
import torch 
from torch import nn 
from transformers import PreTrainedModel


class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaForConditionalGeneration(PreTrainedModel):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.config = config
        
        # Vision model component
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Language model component
        self.language_model = GemmaForCausalLM(config.language_model_config)
        self.vocab_size = self.language_model.config.vocab_size
        
        # Projector to align image features dimensionality with language model hidden size
        self.multi_modal_projector = nn.Linear(config.vision_config.embed_dim, config.language_model_config.hidden_size)
        
        # Handle padding if configured
        self.pad_tokens = config.pad_tokens if hasattr(config, 'pad_tokens') else -1
        
        # Tie weights if language model uses shared embeddings
        self.language_model.tie_weights()
        
    def forward(self, input_ids, pixel_values, attention_mask=None, kv_cache=None):
        # Ensure input dimensions and types are correct
        assert input_ids is not None, "input_ids must be provided"
        assert pixel_values is not None, "pixel_values must be provided"
        
        # Get input embeddings from the language model
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Process image features through the vision model
        selected_image_features = self.vision_tower(pixel_values)
        
        # Project image features to match language model's hidden size
        image_features = self.multi_modal_projector(selected_image_features)
        
        # Merge image features with text embeddings
        input_embeds, attn_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, input_embeddings, input_ids, attention_mask, kv_cache
        )
        
        # Pass merged embeddings to the language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attn_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            kv_cache=kv_cache
        )
        
        return outputs
    
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, kv_cache):
        # Custom function to handle merging logic
        # This is a placeholder; implement specific merging strategy here
        batch_size, seq_len, _ = inputs_embeds.size()
        
        # Simple concatenation strategy (placeholder)
        input_embeds = torch.cat([inputs_embeds, image_features], dim=1)
        if attention_mask is not None:
            extended_attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, image_features.shape[1], dtype=torch.long, device=attention_mask.device)
            ], dim=1)
        else:
            extended_attention_mask = None
        
        position_ids = torch.arange(0, seq_len + image_features.shape[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        return input_embeds, extended_attention_mask, position_ids



class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

