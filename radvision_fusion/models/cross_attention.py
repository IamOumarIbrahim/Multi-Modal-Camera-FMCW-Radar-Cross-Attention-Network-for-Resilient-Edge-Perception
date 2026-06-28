
# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import neural network modules from PyTorch
import torch.nn as nn

# =============================================================================
# SECTION: Spatial Tokenization & Learned Positional Embedding Module
# =============================================================================
class SpatialTokenization(nn.Module):
    # Constructor for tokenization layer given spatial dimensions and D=256
    def __init__(self, TokenInit_SpatialHeight, TokenInit_SpatialWidth, TokenInit_LatentDim=256):
        # Invoke parent constructor
        super(SpatialTokenization, self).__init__()
        # Store spatial sequence length (H * W)
        self.TokenInit_SeqLength = TokenInit_SpatialHeight * TokenInit_SpatialWidth
        # Define learned 1D positional embedding parameters initialized randomly
        self.TokenInit_PosEmbedding = nn.Parameter(
            torch.randn(1, self.TokenInit_SeqLength, TokenInit_LatentDim)
        )

    # Forward pass of spatial tokenization
    def forward(self, TokenForward_FeatureMap):
        # Extract batch size, channels, height, and width of feature map
        TokenForward_Batch, TokenForward_Channels, TokenForward_Height, TokenForward_Width = TokenForward_FeatureMap.shape
        
        # Flatten spatial dimensions [Batch, Channels, H, W] -> [Batch, Channels, H * W]
        TokenForward_Flattened = TokenForward_FeatureMap.flatten(start_dim=2, end_dim=3)
        
        # Transpose to shape [Batch, H * W, Channels] to form token sequence representation
        TokenForward_Sequence = TokenForward_Flattened.transpose(1, 2)
        
        # Add learned positional embeddings to token sequences (broadcasting along batch dimension)
        TokenForward_TokensWithPos = TokenForward_Sequence + self.TokenInit_PosEmbedding
        
        # Return positional-embedded tokens
        return TokenForward_TokensWithPos
