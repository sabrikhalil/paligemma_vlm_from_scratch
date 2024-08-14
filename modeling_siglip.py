import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# Define a function to choose the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is:", device)
 
class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        image_size=224,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_img_tokens: int = None,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.layer_norm_eps = layer_norm_eps
        self.num_channels = num_channels
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.num_img_tokens = num_img_tokens
        self.num_hidden_layers = num_hidden_layers
 
class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super(SiglipVisionModel, self).__init__()
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, x):
        return self.vision_model(x)
 
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
 
        # Initialize embeddings and encoder
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)  # Correctly instantiate the encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
 
    def forward(self, pixel_values):
        # Convert image to patches and generate embeddings
        hidden_states = self.embeddings(pixel_values)
        
        # Pass embeddings through the encoder
        last_hidden_state = self.encoder(hidden_states)
        
        # Apply post layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
 
 
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        )
        
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        self.position_embeddings = nn.Embedding(self.num_patches, config.hidden_size)
        
        self.register_buffer("position_ids", torch.arange(self.num_patches).unsqueeze(0))
 
    def forward(self, pixel_values):
        # Convert image to patches and then to embeddings
        embeddings = self.patch_embeddings(pixel_values)  # [batch_size, hidden_size, num_patches_x, num_patches_y]
        
        # Flatten the output from 2D (patch grid) to 1D (sequence of patches)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_size]
 
        # Expand the position IDs to match the batch size and retrieve corresponding embeddings
        position_embeddings = self.position_embeddings(self.position_ids.expand(embeddings.size(0), -1))  # [batch_size, num_patches, hidden_size]
        
        # Add the positional embeddings to the patch embeddings
        embeddings = embeddings + position_embeddings  # [batch_size, num_patches, hidden_size]
 
        return embeddings
    
 
 
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SiglipEncoderLayer, self).__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
 
    def forward(self, x):
        # Self-attention part
        attn_output = self.self_attn(x)  # [batch_size, num_patches, hidden_size]
        # Apply first layer normalization (post-attention)
        out1 = self.layer_norm1(x + attn_output)  # [batch_size, num_patches, hidden_size]
 
        # MLP part
        mlp_output = self.mlp(out1)  # [batch_size, num_patches, hidden_size]
        # Apply second layer normalization (post-MLP)
        out2 = self.layer_norm2(out1 + mlp_output)  # [batch_size, num_patches, hidden_size]
 
        return out2
    
 
 
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super(SiglipMLP, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()  # GELU could also be used
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
 
    def forward(self, x):
        x = self.fc1(x)  # [batch_size, num_patches, intermediate_size]
        x = self.activation(x)
        x = self.fc2(x)  # [batch_size, num_patches, hidden_size]
        return x
    
 
 
class SiglipAttention(nn.Module):
    def __init__(self, config):
        super(SiglipAttention, self).__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        # Projection layers for queries, keys, and values
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        # Output projection layer
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Optional dropout layer
        self.dropout = nn.Dropout(config.attention_dropout)
 
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        
        # Linear projections
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        
        # Attention mechanism (scaled dot-product attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_length, seq_length]
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        # Combine the scores with the value vectors
        attn_output = torch.matmul(scores, v)  # [batch_size, num_heads, seq_length, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)  # Combine heads and project
        
        # Final output projection
        output = self.out_proj(attn_output)  # [batch_size, seq_length, embed_dim]
        
        return output
 
 
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config):
        super(SiglipVisionEncoder, self).__init__()
        self.layer_count = config.num_hidden_layers  # Number of encoder layers
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(self.layer_count)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Optional final normalization
 
    def forward(self, hidden_states):
        # Process each encoder layer
        for layer in self.layers:
            hidden_states = layer(hidden_states)  # [batch_size, num_patches, hidden_size]
 
        # Apply final layer normalization
        encoded_output = self.layer_norm(hidden_states)  # [batch_size, num_patches, hidden_size]
 
        return encoded_output
 