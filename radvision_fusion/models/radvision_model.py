# =============================================================================
# STUDENT-CODING-AGENT NOTICE:
# Style rules, including line-by-line comments and prefixed variable names,
# are applied internally in this model integration file.
# =============================================================================

# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import base neural network class from PyTorch
import torch.nn as nn
# Import subcomponent classes from their respective local modular files
from radvision_fusion.models.vision_encoder import VisionEncoder
from radvision_fusion.models.radar_encoder import RadarEncoder
from radvision_fusion.models.cross_attention import SpatialTokenization
from radvision_fusion.models.detection_head import VisionDetectionHead, RadarDetectionHead

# =============================================================================
# SECTION: Unified RadVision-Fusion Model Class
# =============================================================================
class RadVisionFusionModel(nn.Module):
    # Constructor mapping model components to shared latent space (D = 256)
    def __init__(self, ModelInit_LatentDim=256, ModelInit_Mode="fused"):
        # Invoke parent constructor
        super(RadVisionFusionModel, self).__init__()
        
        # Save shared embedding dimension size
        self.ModelInit_LatentDim = ModelInit_LatentDim
        # Save configuration execution mode ("fused", "vision_only", or "radar_only")
        self.ModelInit_Mode = ModelInit_Mode
        
        # Instantiate Vision Encoder Branch
        self.ModelInit_VisionEncoder = VisionEncoder(VisionInit_LatentDim=ModelInit_LatentDim)
        
        # Instantiate Radar Encoder Branch for Range-Doppler maps
        self.ModelInit_RadarEncoderRD = RadarEncoder(RadarInit_LatentDim=ModelInit_LatentDim)
        
        # Instantiate Radar Encoder Branch for Range-Angle maps
        self.ModelInit_RadarEncoderRA = RadarEncoder(RadarInit_LatentDim=ModelInit_LatentDim)
        
        # Tokenizer for Camera Frame features (ResNet-18 layer 4 output shape is 7x7)
        self.ModelInit_VisionTokenizer = SpatialTokenization(
            TokenInit_SpatialHeight=7, 
            TokenInit_SpatialWidth=7, 
            TokenInit_LatentDim=ModelInit_LatentDim
        )
        
        # Tokenizer for Radar Range-Doppler features (RD custom CNN output shape is 16x4)
        self.ModelInit_RadarRDTokenizer = SpatialTokenization(
            TokenInit_SpatialHeight=16, 
            TokenInit_SpatialWidth=4, 
            TokenInit_LatentDim=ModelInit_LatentDim
        )
        
        # Tokenizer for Radar Range-Angle features (RA custom CNN output shape is 16x16)
        self.ModelInit_RadarRATokenizer = SpatialTokenization(
            TokenInit_SpatialHeight=16, 
            TokenInit_SpatialWidth=16, 
            TokenInit_LatentDim=ModelInit_LatentDim
        )
        
        # Multi-Head Cross-Attention Layer (Camera Queries Q, Radar Keys K & Values V)
        self.ModelInit_CrossAttention = nn.MultiheadAttention(
            embed_dim=ModelInit_LatentDim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Feed-Forward Network applied token-wise to query tokens
        self.ModelInit_FFN = nn.Sequential(
            nn.Linear(ModelInit_LatentDim, ModelInit_LatentDim * 2),
            nn.ReLU(),
            nn.Linear(ModelInit_LatentDim * 2, ModelInit_LatentDim)
        )
        
        # Layer Normalization layers for attention blocks
        self.ModelInit_Norm1 = nn.LayerNorm(ModelInit_LatentDim)
        self.ModelInit_Norm2 = nn.LayerNorm(ModelInit_LatentDim)
        
        # Instantiate Vision / Fused detection head predicting boxes and classes
        self.ModelInit_VisionDetHead = VisionDetectionHead(
            HeadInit_NumTokens=49, 
            HeadInit_LatentDim=ModelInit_LatentDim
        )
        
        # Instantiate Radar-Only detection head predicting boxes and classes
        self.ModelInit_RadarDetHead = RadarDetectionHead(
            HeadInit_LatentDim=ModelInit_LatentDim
        )

    # Forward pass mapping inputs to class probabilities and bounding box bounds
    def forward(self, ModelForward_CamTensor, ModelForward_RDTensor, ModelForward_RATensor):
        # Handle Radar-Only Mode (Camera path bypassed completely)
        if self.ModelInit_Mode == "radar_only":
            # Process Range-Doppler map through corresponding encoder
            ModelForward_RDFeatures = self.ModelInit_RadarEncoderRD(ModelForward_RDTensor)
            # Process Range-Angle map through corresponding encoder
            ModelForward_RAFeatures = self.ModelInit_RadarEncoderRA(ModelForward_RATensor)
            # Tokenize Range-Doppler features
            ModelForward_RDTokens = self.ModelInit_RadarRDTokenizer(ModelForward_RDFeatures)
            # Tokenize Range-Angle features
            ModelForward_RATokens = self.ModelInit_RadarRATokenizer(ModelForward_RAFeatures)
            # Combine all radar tokens along sequence axis
            ModelForward_RadarTokensCombined = torch.cat([ModelForward_RDTokens, ModelForward_RATokens], dim=1)
            # Perform Global Average Pooling over token sequence: [Batch, 320, 256] -> [Batch, 256]
            ModelForward_PooledRadar = ModelForward_RadarTokensCombined.mean(dim=1)
            # Generate predictions using Radar-Only head
            ModelForward_ClassLogit, ModelForward_BBoxPred = self.ModelInit_RadarDetHead(ModelForward_PooledRadar)
            # Return prediction outputs and mock attention weights
            return ModelForward_ClassLogit, ModelForward_BBoxPred, None

        # Handle Vision-Only Mode (Radar path bypassed completely)
        elif self.ModelInit_Mode == "vision_only":
            # Extract features for Camera frames
            ModelForward_CamFeatures = self.ModelInit_VisionEncoder(ModelForward_CamTensor)
            # Convert camera features to tokens
            ModelForward_CamTokens = self.ModelInit_VisionTokenizer(ModelForward_CamFeatures)
            # Bypass cross-attention (Residual identity + FFN)
            ModelForward_FFNResult = self.ModelInit_FFN(ModelForward_CamTokens)
            ModelForward_FusedTokens = self.ModelInit_Norm2(ModelForward_CamTokens + ModelForward_FFNResult)
            # Flatten attended tokens: [Batch, 49, 256] -> [Batch, 49 * 256]
            ModelForward_FlattenedTokens = ModelForward_FusedTokens.flatten(start_dim=1)
            # Generate predictions using Vision head
            ModelForward_ClassLogit, ModelForward_BBoxPred = self.ModelInit_VisionDetHead(ModelForward_FlattenedTokens)
            # Return prediction outputs and mock attention weights
            return ModelForward_ClassLogit, ModelForward_BBoxPred, None

        # Standard Multi-Modal Fused Mode
        else:
            # 1. Feature extraction using encoders
            ModelForward_CamFeatures = self.ModelInit_VisionEncoder(ModelForward_CamTensor)
            ModelForward_RDFeatures = self.ModelInit_RadarEncoderRD(ModelForward_RDTensor)
            ModelForward_RAFeatures = self.ModelInit_RadarEncoderRA(ModelForward_RATensor)
            
            # 2. Convert features into token sequences and add learned positional embeddings
            ModelForward_CamTokens = self.ModelInit_VisionTokenizer(ModelForward_CamFeatures)
            ModelForward_RDTokens = self.ModelInit_RadarRDTokenizer(ModelForward_RDFeatures)
            ModelForward_RATokens = self.ModelInit_RadarRATokenizer(ModelForward_RAFeatures)
            
            # 3. Concatenate Range-Doppler and Range-Angle tokens along the sequence dimension
            ModelForward_RadarTokensCombined = torch.cat(
                [ModelForward_RDTokens, ModelForward_RATokens], 
                dim=1
            )
            
            # 4. Multi-Head Cross-Attention (Query = Camera, Key/Value = Combined Radar)
            ModelForward_AttnOutput, ModelForward_AttnWeights = self.ModelInit_CrossAttention(
                query=ModelForward_CamTokens,
                key=ModelForward_RadarTokensCombined,
                value=ModelForward_RadarTokensCombined
            )
            
            # 5. Residual Connection + Layer Normalization for attention output
            ModelForward_AttendedCam = self.ModelInit_Norm1(ModelForward_CamTokens + ModelForward_AttnOutput)
            
            # 6. Apply FFN + residual connection + layer normalization
            ModelForward_FFNResult = self.ModelInit_FFN(ModelForward_AttendedCam)
            ModelForward_FusedTokens = self.ModelInit_Norm2(ModelForward_AttendedCam + ModelForward_FFNResult)
            
            # 7. Flatten attended tokens: [Batch, 49, 256] -> [Batch, 49 * 256]
            ModelForward_FlattenedTokens = ModelForward_FusedTokens.flatten(start_dim=1)
            
            # 8. Feed flattened features into shared MLP layers
            ModelForward_ClassLogit, ModelForward_BBoxPred = self.ModelInit_VisionDetHead(ModelForward_FlattenedTokens)
            
            # Return class logit, bounding boxes, and attention weights for interpretability/visualization
            return ModelForward_ClassLogit, ModelForward_BBoxPred, ModelForward_AttnWeights
