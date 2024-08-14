from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
import torch
from torch import nn
from typing import Optional, Tuple, List


class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

 
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
 
 
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        
        # Vision model component
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Language model component
        self.language_model = GemmaForCausalLM(config.text_config)
        self.vocab_size = config.vocab_size
        
        print("hidden size: ", config.vision_config.hidden_size)
        print("projection_dim : ", config.vision_config.projection_dim) 
        # Projector to align image features dimensionality with language model hidden size
        self.multi_modal_projector = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

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
        
        print("selected_image_features shape:", selected_image_features.shape)  # Should be [batch, 196, 768]
        print("multi_modal_projector in_features:", self.multi_modal_projector.in_features)  # Should be 768
        print("multi_modal_projector out_features:", self.multi_modal_projector.out_features)  # Should be 2048


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
    
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, kv_cache: Optional[KVCache] = None):
        batch_size, seq_len, embed_dim = inputs_embeds.size()
 
        # Create a final embedding tensor that starts with input embeddings
        final_embedding = inputs_embeds.clone()
 
        # Create masks for text and image positions
        text_mask = (input_ids != self.config.image_token_index)
        image_mask = (input_ids == self.config.image_token_index)
 
        # Ensure image_features are scaled or transformed as needed to match embedding dimension
        # This step assumes image_features are processed to be placed correctly
        scaled_image_features = self.multi_modal_projector(image_features)
 
        # Insert image embeddings at positions marked by image_mask
        for i in range(batch_size):
            final_embedding[i, image_mask[i], :] = scaled_image_features[i, :image_mask[i].sum(), :]
 
        # Optionally handle attention mask
        if attention_mask is not None:
            # Extend attention_mask to cover image tokens
            extended_attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, image_features.shape[1], dtype=torch.long, device=attention_mask.device)
            ], dim=1)
        else:
            extended_attention_mask = None
 
        # Create position IDs for the entire sequence
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
 
        return final_embedding, extended_attention_mask, position_ids
 
 
 
 
 
 
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super(GemmaForCausalLM, self).__init__()
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
 
        # Linear layer to map hidden states to vocabulary size for output
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
 
        # Tie weights between input embeddings and the output layer
        self.tie_weights()
 
    def tie_weights(self):
        """ Ensure the weights of the input embeddings and the output lm_head are the same. """
        self.lm_head.weight = self.model.embed_tokens.weight
 
    def get_input_embeddings(self):
        """ Returns the embeddings tensor from the model. """
        return self.model.embed_tokens
 
    def forward(self, attention_mask, position_ids, inputs_embeds, kv_cache=None):
        """ Process input through the model and return the logits. """
        # Process inputs through the model
        hidden_states = self.model(inputs_embeds, attention_mask, position_ids, kv_cache)
 
        # Apply the language modeling head to generate logits
        logits = self.lm_head(hidden_states)
        return logits
 
 
 
 
class GemmaModel(nn.Module):
    def __init__(self, config):
        super(GemmaModel, self).__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
 
        # Embedding layer
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
 
        # List of decoder layers
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
 
        # Normalization layer
        self.norm = GemmaRMSNorm(config.hidden_size)
 
    def forward(self, inputs_embeds, attention_mask, position_ids, kv_cache=None):
        """ Forward pass through the model. """
        # Normalize input embeddings
        scale = torch.tensor(self.config.hidden_size ** 0.5, device=inputs_embeds.device)
        normalized_embeds = inputs_embeds * scale
 
        # Pass through each decoder layer
        hidden_states = normalized_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, kv_cache)
 
        # Apply final normalization
        outputs = self.norm(hidden_states)
        return outputs
 
 
 
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(GemmaRMSNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))
 
    def forward(self, x):
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized_x = x / rms
        return self.scale * normalized_x
 
 
 
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super(GemmaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size)
 
    def forward(self, hidden_states, attn_mask, pos_ids, kv_cache=None):
        # Apply input layer normalization
        norm_states = self.input_layernorm(hidden_states)
 
        # Self-attention
        attn_output = self.self_attn(norm_states, attn_mask, pos_ids, kv_cache)
        # First residual connection
        hidden_states = hidden_states + attn_output
 
        # Post-attention layer normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
 
        # MLP
        mlp_output = self.mlp(hidden_states)
        # Second residual connection
        hidden_states = hidden_states + mlp_output
 
        return hidden_states
 
 
 
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super(GemmaMLP, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediary_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediary_size)
        self.down_proj = nn.Linear(self.intermediary_size, self.hidden_size)
 
    def forward(self, x):
        # Gated activation unit
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
 
 
 
 
 
 
class KVCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []
 
    def update(self, keys, values, layer_idx):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(keys)
            self.value_cache.append(values)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], keys], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], values], dim=-2)
 
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
 
    def length(self):
        return 0 if len(self.key_cache) == 0 else self.key_cache[0].shape[-2]
 
 
class GemmaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
 
        # Initialize projections for queries, keys, and values
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_key_value_heads * self.head_dim, config.hidden_size)
 
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, config.max_position_embeddings, base=config.rope_theta)
 
    def forward(self, hidden_states, attention_mask, position_ids, kv_cache=None):
        bs, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
 
        # Apply rotary position embeddings
        q, k = self.apply_rotary_pos_emb(q, k, position_ids)
 
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)
 
        # Reshape for multi-head attention
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, -1, self.num_key_value_heads, self.head_dim)
        v = v.view(bs, -1, self.num_key_value_heads, self.head_dim)
 
        # Compute scaled dot-product attention
        attn_weights = torch.einsum('bnhd,bmhd->bhnm', q, k) * (self.head_dim ** -0.5)
        if attention_mask is not None:
            attn_weights += attention_mask[:, None, None, :]
 
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
 
        # Attend to the values
        attn_output = torch.einsum('bhnm,bmhd->bnhd', attn_weights, v)
        attn_output = attn_output.reshape(bs, seq_len, self.num_key_value_heads * self.head_dim)
 
        # Final projection
        attn_output = self.o_proj(attn_output)
 
        return attn_output, attn_weights
 
    def apply_rotary_pos_emb(self, q, k, position_ids):
        cos, sin = self.rotary_emb(position_ids)
        return self.rotate_half(q, cos, sin), self.rotate_half(k, cos, sin)
 
    def rotate_half(self, x, cos, sin):
        x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
        return torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1)
 
 
 
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
 
        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
 
        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
 
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
 
