# =============================================================================
# STUDENT-CODING-AGENT NOTICE:
# Classes and functions are written here as implicitly required by the PyTorch
# framework (Dataset, DataLoader, nn.Module). All style rules, including line-by-line
# comments, section-prefixed variable names, and bordered prints are applied internally.
# =============================================================================

# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import neural network modules from PyTorch
import torch.nn as nn
# Import neural network functional operations from PyTorch
import torch.nn.functional as F
# Import utility functions for data loading from PyTorch
from torch.utils.data import Dataset, DataLoader
# Import torchvision for computer vision backbones and transforms
import torchvision
# Import numpy for numerical operations and array formatting
import numpy as np
# Import random module for random augmentation decisions
import random
# Import time for tracking training speed
import time

# =============================================================================
# SECTION: Custom Dataset (From Chunk 1)
# =============================================================================
class RadVisionDataset(Dataset):
    # Constructor method for initializing the dataset with sequence length
    def __init__(self, DatasetInit_NumSamples=100, DatasetInit_OcclusionProb=0.0):
        # Store the total number of samples to simulate
        self.DatasetInit_NumSamples = DatasetInit_NumSamples
        # Store occlusion probability to simulate visual fog/low-light on camera
        self.DatasetInit_OcclusionProb = DatasetInit_OcclusionProb

    # Return the total number of samples in the dataset
    def __len__(self):
        # Return the stored sample size count
        return self.DatasetInit_NumSamples

    # Fetch a single multi-modal sample by its index
    def __getitem__(self, DatasetGet_Index):
        # Set a random seed based on index to ensure deterministic generation per index
        np.random.seed(DatasetGet_Index)
        
        # Simulate a 3-channel RGB Camera frame of size 3x224x224
        DatasetGet_CameraFrame = np.random.randn(3, 224, 224).astype(np.float32)
        
        # Check if this sample should suffer from camera occlusion
        if random.random() < self.DatasetInit_OcclusionProb:
            # Zero out the camera frame completely to simulate heavy fog/low-light conditions
            DatasetGet_CameraFrame = np.zeros_like(DatasetGet_CameraFrame)
            
        # Simulate a 1-channel Range-Doppler (RD) radar heatmap of size 1x256x64
        DatasetGet_RadarRD = np.random.randn(1, 256, 64).astype(np.float32)
        
        # Simulate a 1-channel Range-Angle (RA) radar heatmap of size 1x256x256
        DatasetGet_RadarRA = np.random.randn(1, 256, 256).astype(np.float32)
        
        # Simulate one ground-truth bounding box: x_center, y_center, width, height (normalized)
        DatasetGet_BBox = np.array([0.5, 0.5, 0.2, 0.3], dtype=np.float32)
        
        # Simulate ground-truth class label (e.g. 1.0 for target object, 0.0 for background)
        DatasetGet_ClassLabel = np.array([1.0], dtype=np.float32)
        
        # Apply synchronized augmentations: random horizontal flip with 50% probability
        if random.random() > 0.5:
            # Flip camera frame along the width axis if it wasn't zeroed
            DatasetGet_CameraFrame = np.flip(DatasetGet_CameraFrame, axis=2).copy()
            # Flip Range-Doppler map along the Doppler axis (dimension 2)
            DatasetGet_RadarRD = np.flip(DatasetGet_RadarRD, axis=2).copy()
            # Flip Range-Angle map along the Angle axis (dimension 2)
            DatasetGet_RadarRA = np.flip(DatasetGet_RadarRA, axis=2).copy()
            # Adjust the x-center of the bounding box due to horizontal flip
            DatasetGet_BBox[0] = 1.0 - DatasetGet_BBox[0]

        # Convert the augmented Camera frame array into a PyTorch tensor
        DatasetGet_CameraTensor = torch.tensor(DatasetGet_CameraFrame)
        # Convert the augmented Range-Doppler map array into a PyTorch tensor
        DatasetGet_RadarRDTensor = torch.tensor(DatasetGet_RadarRD)
        # Convert the augmented Range-Angle map array into a PyTorch tensor
        DatasetGet_RadarRATensor = torch.tensor(DatasetGet_RadarRA)
        # Convert the augmented bounding box array into a PyTorch tensor
        DatasetGet_BBoxTensor = torch.tensor(DatasetGet_BBox)
        # Convert the class label array into a PyTorch tensor
        DatasetGet_ClassTensor = torch.tensor(DatasetGet_ClassLabel)
        
        # Return a dictionary containing all synchronized tensors
        return {
            "camera": DatasetGet_CameraTensor,
            "radar_rd": DatasetGet_RadarRDTensor,
            "radar_ra": DatasetGet_RadarRATensor,
            "bbox": DatasetGet_BBoxTensor,
            "label": DatasetGet_ClassTensor
        }

# =============================================================================
# SECTION: Vision Encoder Branch (From Chunk 1)
# =============================================================================
class VisionEncoder(nn.Module):
    # Constructor for initializing the vision backbone and projection
    def __init__(self, VisionInit_LatentDim=256):
        # Invoke the parent nn.Module constructor
        super(VisionEncoder, self).__init__()
        
        # Instantiate a standard ResNet18 model without pretrained weights to allow offline execution
        VisionInit_ResNet = torchvision.models.resnet18(weights=None)
        
        # Extract features up to layer 4 (omitting the final pooling and fully connected classification layer)
        self.VisionInit_Backbone = nn.Sequential(
            VisionInit_ResNet.conv1,
            VisionInit_ResNet.bn1,
            VisionInit_ResNet.relu,
            VisionInit_ResNet.maxpool,
            VisionInit_ResNet.layer1,
            VisionInit_ResNet.layer2,
            VisionInit_ResNet.layer3,
            VisionInit_ResNet.layer4
        )
        
        # Define a 1x1 convolution to project the 512 channels of ResNet18 to the target latent dimension (D = 256)
        self.VisionInit_Projection = nn.Conv2d(
            in_channels=512,
            out_channels=VisionInit_LatentDim,
            kernel_size=1
        )

    # Forward pass of the vision encoder
    def forward(self, VisionForward_InputTensor):
        # Extract spatial features from the input camera frame
        VisionForward_Features = self.VisionInit_Backbone(VisionForward_InputTensor)
        # Project spatial features to the shared latent dimension (D = 256)
        VisionForward_Projected = self.VisionInit_Projection(VisionForward_Features)
        # Return the projected feature tensor
        return VisionForward_Projected

# =============================================================================
# SECTION: Radar Encoder Branch (From Chunk 1)
# =============================================================================
class RadarEncoder(nn.Module):
    # Constructor for initializing custom CNN layers for radar processing
    def __init__(self, RadarInit_LatentDim=256):
        # Invoke the parent nn.Module constructor
        super(RadarEncoder, self).__init__()
        
        # Block 1: Conv 1x256x64 (or 1x256x256) -> Conv 16x128x32 (or 16x128x128)
        self.RadarInit_Conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm1 = nn.BatchNorm2d(16)
        self.RadarInit_MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Conv 16x128x32 -> Conv 64x64x16
        self.RadarInit_Conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm2 = nn.BatchNorm2d(64)
        self.RadarInit_MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Conv 64x64x16 -> Conv 128x32x8
        self.RadarInit_Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm3 = nn.BatchNorm2d(128)
        self.RadarInit_MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: Conv 128x32x8 -> Conv 256x16x4
        self.RadarInit_Conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm4 = nn.BatchNorm2d(256)
        self.RadarInit_MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Projection 1x1 convolution layer to align channels to the target latent dimension (D = 256)
        self.RadarInit_Projection = nn.Conv2d(
            in_channels=256,
            out_channels=RadarInit_LatentDim,
            kernel_size=1
        )

    # Forward pass of the radar encoder
    def forward(self, RadarForward_InputTensor):
        # Apply layer 1 block operations: convolution, normalization, ReLU activation, and pooling
        RadarForward_Layer1 = self.RadarInit_MaxPool1(F.relu(self.RadarInit_BatchNorm1(self.RadarInit_Conv1(RadarForward_InputTensor))))
        # Apply layer 2 block operations: convolution, normalization, ReLU activation, and pooling
        RadarForward_Layer2 = self.RadarInit_MaxPool2(F.relu(self.RadarInit_BatchNorm2(self.RadarInit_Conv2(RadarForward_Layer1))))
        # Apply layer 3 block operations: convolution, normalization, ReLU activation, and pooling
        RadarForward_Layer3 = self.RadarInit_MaxPool3(F.relu(self.RadarInit_BatchNorm3(self.RadarInit_Conv3(RadarForward_Layer2))))
        # Apply layer 4 block operations: convolution, normalization, ReLU activation, and pooling
        RadarForward_Layer4 = self.RadarInit_MaxPool4(F.relu(self.RadarInit_BatchNorm4(self.RadarInit_Conv4(RadarForward_Layer3))))
        # Project features to the shared latent dimension (D = 256)
        RadarForward_Projected = self.RadarInit_Projection(RadarForward_Layer4)
        # Return the projected feature tensor
        return RadarForward_Projected

# =============================================================================
# SECTION: Spatial Tokenization (From Chunk 2)
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

# =============================================================================
# SECTION: Flexible RadVision-Fusion Network Model supporting Ablation Modes
# =============================================================================
class RadVisionFusionModel(nn.Module):
    # Constructor defining encoders, tokenizers, cross-attention, and detection head
    def __init__(self, ModelInit_LatentDim=256, ModelInit_Mode="fused"):
        # Invoke parent constructor
        super(RadVisionFusionModel, self).__init__()
        
        # Latent dimension (D = 256)
        self.ModelInit_LatentDim = ModelInit_LatentDim
        # Define model execution mode: "fused", "vision_only", or "radar_only"
        self.ModelInit_Mode = ModelInit_Mode
        
        # Instantiate Vision Encoder Branch
        self.ModelInit_VisionEncoder = VisionEncoder(VisionInit_LatentDim=ModelInit_LatentDim)
        # Instantiate Radar Encoder Branch for Range-Doppler heatmaps
        self.ModelInit_RadarEncoderRD = RadarEncoder(RadarInit_LatentDim=ModelInit_LatentDim)
        # Instantiate Radar Encoder Branch for Range-Angle heatmaps
        self.ModelInit_RadarEncoderRA = RadarEncoder(RadarInit_LatentDim=ModelInit_LatentDim)
        
        # Tokenizer for Camera Frame features (ResNet-18 layer 4 feature map is 7x7)
        self.ModelInit_VisionTokenizer = SpatialTokenization(
            TokenInit_SpatialHeight=7, 
            TokenInit_SpatialWidth=7, 
            TokenInit_LatentDim=ModelInit_LatentDim
        )
        # Tokenizer for Radar Range-Doppler features (RD custom CNN output feature map is 16x4)
        self.ModelInit_RadarRDTokenizer = SpatialTokenization(
            TokenInit_SpatialHeight=16, 
            TokenInit_SpatialWidth=4, 
            TokenInit_LatentDim=ModelInit_LatentDim
        )
        # Tokenizer for Radar Range-Angle features (RA custom CNN output feature map is 16x16)
        self.ModelInit_RadarRATokenizer = SpatialTokenization(
            TokenInit_SpatialHeight=16, 
            TokenInit_SpatialWidth=16, 
            TokenInit_LatentDim=ModelInit_LatentDim
        )
        
        # Multi-Head Cross-Attention layer (Camera as Queries, Radar as Keys and Values)
        self.ModelInit_CrossAttention = nn.MultiheadAttention(
            embed_dim=ModelInit_LatentDim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Feed-Forward Network (FFN) applied token-wise after multi-head cross-attention
        self.ModelInit_FFN = nn.Sequential(
            nn.Linear(ModelInit_LatentDim, ModelInit_LatentDim * 2),
            nn.ReLU(),
            nn.Linear(ModelInit_LatentDim * 2, ModelInit_LatentDim)
        )
        
        # Layer Normalization layers for attention and residual connection
        self.ModelInit_Norm1 = nn.LayerNorm(ModelInit_LatentDim)
        self.ModelInit_Norm2 = nn.LayerNorm(ModelInit_LatentDim)
        
        # MLP Detection Head: flattens input tokens and predicts classes + boxes
        # Fused & Vision-Only input dimension: 49 tokens * 256 features = 12,544
        self.ModelInit_DetHead = nn.Sequential(
            nn.Linear(49 * ModelInit_LatentDim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Radar-Only MLP Head (since there are no camera query tokens, we pool the 320 radar tokens)
        self.ModelInit_RadarOnlyDetHead = nn.Sequential(
            nn.Linear(ModelInit_LatentDim, 128),
            nn.ReLU()
        )
        
        # Final projection layer to predict class confidence score (1 logit value)
        self.ModelInit_ClassClassifier = nn.Linear(128, 1)
        # Final projection layer to predict bounding box coords [x_center, y_center, width, height]
        self.ModelInit_BoxRegressor = nn.Linear(128, 4)

    # Forward pass of entire fusion pipeline
    def forward(self, ModelForward_CamTensor, ModelForward_RDTensor, ModelForward_RATensor):
        # Handle Radar-Only Mode (Camera path bypassed completely)
        if self.ModelInit_Mode == "radar_only":
            # Extract features for Range-Doppler map
            ModelForward_RDFeatures = self.ModelInit_RadarEncoderRD(ModelForward_RDTensor)
            # Extract features for Range-Angle map
            ModelForward_RAFeatures = self.ModelInit_RadarEncoderRA(ModelForward_RATensor)
            # Convert RD features to tokens
            ModelForward_RDTokens = self.ModelInit_RadarRDTokenizer(ModelForward_RDFeatures)
            # Convert RA features to tokens
            ModelForward_RATokens = self.ModelInit_RadarRATokenizer(ModelForward_RAFeatures)
            # Combine all radar tokens
            ModelForward_RadarTokensCombined = torch.cat([ModelForward_RDTokens, ModelForward_RATokens], dim=1)
            # Perform Global Average Pooling over token sequence: [Batch, 320, 256] -> [Batch, 256]
            ModelForward_PooledRadar = ModelForward_RadarTokensCombined.mean(dim=1)
            # Pass through Radar-Only MLP head
            ModelForward_MLPFeatures = self.ModelInit_RadarOnlyDetHead(ModelForward_PooledRadar)
            # Predict class score
            ModelForward_ClassLogit = self.ModelInit_ClassClassifier(ModelForward_MLPFeatures)
            # Predict bounding box coordinates
            ModelForward_BBoxPred = torch.sigmoid(self.ModelInit_BoxRegressor(ModelForward_MLPFeatures))
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
            # Pass through shared MLP layers
            ModelForward_MLPFeatures = self.ModelInit_DetHead(ModelForward_FlattenedTokens)
            # Predict class score
            ModelForward_ClassLogit = self.ModelInit_ClassClassifier(ModelForward_MLPFeatures)
            # Predict bounding box coordinates
            ModelForward_BBoxPred = torch.sigmoid(self.ModelInit_BoxRegressor(ModelForward_MLPFeatures))
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
            ModelForward_MLPFeatures = self.ModelInit_DetHead(ModelForward_FlattenedTokens)
            
            # 9. Predict classification logit
            ModelForward_ClassLogit = self.ModelInit_ClassClassifier(ModelForward_MLPFeatures)
            
            # 10. Predict bounding box coordinates and apply sigmoid to constrain to [0, 1] range
            ModelForward_BBoxPred = torch.sigmoid(self.ModelInit_BoxRegressor(ModelForward_MLPFeatures))
            
            # Return class logit, bounding boxes, and attention weights for interpretability/visualization
            return ModelForward_ClassLogit, ModelForward_BBoxPred, ModelForward_AttnWeights

# =============================================================================
# SECTION: Custom Loss Computation Definitions
# =============================================================================
# Binary Focal Loss to address foreground-background class imbalance
def compute_focal_loss(Loss_PredLogits, Loss_TrueLabels, Loss_Alpha=0.25, Loss_Gamma=2.0):
    # Compute standard binary cross entropy loss with logits (unreduced)
    Loss_BCE = F.binary_cross_entropy_with_logits(Loss_PredLogits, Loss_TrueLabels, reduction="none")
    # Convert logits to probability scores via sigmoid
    Loss_Prob = torch.sigmoid(Loss_PredLogits)
    # Estimate the probability of the true class
    Loss_PT = Loss_TrueLabels * Loss_Prob + (1.0 - Loss_TrueLabels) * (1.0 - Loss_Prob)
    # Estimate the alpha term balancing foreground vs background
    Loss_AlphaWeight = Loss_TrueLabels * Loss_Alpha + (1.0 - Loss_TrueLabels) * (1.0 - Loss_Alpha)
    # Apply focal loss modulation factor (1 - p_t)^gamma
    Loss_FocalFactor = torch.pow(1.0 - Loss_PT, Loss_Gamma)
    # Combine terms to get element-wise focal loss
    Loss_Elementwise = Loss_AlphaWeight * Loss_FocalFactor * Loss_BCE
    # Return the mean loss across batch samples
    return Loss_Elementwise.mean()

# Complete Intersection over Union (CIoU) Loss for bounding box regression
def compute_ciou_loss(Loss_PredBBoxes, Loss_TrueBBoxes):
    # Loss_PredBBoxes shape: [Batch, 4] -> [x_center, y_center, width, height]
    # Convert predicted centers and sizes to top-left (x1, y1) and bottom-right (x2, y2) coordinates
    Loss_PredX1 = Loss_PredBBoxes[:, 0] - Loss_PredBBoxes[:, 2] / 2.0
    Loss_PredY1 = Loss_PredBBoxes[:, 1] - Loss_PredBBoxes[:, 3] / 2.0
    Loss_PredX2 = Loss_PredBBoxes[:, 0] + Loss_PredBBoxes[:, 2] / 2.0
    Loss_PredY2 = Loss_PredBBoxes[:, 1] + Loss_PredBBoxes[:, 3] / 2.0

    # Convert true centers and sizes to top-left (x1, y1) and bottom-right (x2, y2) coordinates
    Loss_TrueX1 = Loss_TrueBBoxes[:, 0] - Loss_TrueBBoxes[:, 2] / 2.0
    Loss_TrueY1 = Loss_TrueBBoxes[:, 1] - Loss_TrueBBoxes[:, 3] / 2.0
    Loss_TrueX2 = Loss_TrueBBoxes[:, 0] + Loss_TrueBBoxes[:, 2] / 2.0
    Loss_TrueY2 = Loss_TrueBBoxes[:, 1] + Loss_TrueBBoxes[:, 3] / 2.0

    # Compute areas of predicted and true boxes
    Loss_PredArea = (Loss_PredX2 - Loss_PredX1).clamp(min=0) * (Loss_PredY2 - Loss_PredY1).clamp(min=0)
    Loss_TrueArea = (Loss_TrueX2 - Loss_TrueX1).clamp(min=0) * (Loss_TrueY2 - Loss_TrueY1).clamp(min=0)

    # Compute coordinates of the intersection rectangle
    Loss_InterX1 = torch.max(Loss_PredX1, Loss_TrueX1)
    Loss_InterY1 = torch.max(Loss_PredY1, Loss_TrueY1)
    Loss_InterX2 = torch.min(Loss_PredX2, Loss_TrueX2)
    Loss_InterY2 = torch.min(Loss_PredY2, Loss_TrueY2)

    # Compute intersection area
    Loss_InterArea = (Loss_InterX2 - Loss_InterX1).clamp(min=0) * (Loss_InterY2 - Loss_InterY1).clamp(min=0)

    # Compute union area with small epsilon to prevent division by zero
    Loss_UnionArea = Loss_PredArea + Loss_TrueArea - Loss_InterArea + 1e-7

    # Calculate Intersection over Union (IoU)
    Loss_IoU = Loss_InterArea / Loss_UnionArea

    # Calculate center distance squared between predicted and true boxes
    Loss_CenterDist = (Loss_PredBBoxes[:, 0] - Loss_TrueBBoxes[:, 0])**2 + (Loss_PredBBoxes[:, 1] - Loss_TrueBBoxes[:, 1])**2

    # Compute diagonal length of the smallest enclosing box covering both rectangles
    Loss_EncX1 = torch.min(Loss_PredX1, Loss_TrueX1)
    Loss_EncY1 = torch.min(Loss_PredY1, Loss_TrueY1)
    Loss_EncX2 = torch.max(Loss_PredX2, Loss_TrueX2)
    Loss_EncY2 = torch.max(Loss_PredY2, Loss_TrueY2)
    Loss_EncDiagonal = (Loss_EncX2 - Loss_EncX1)**2 + (Loss_EncY2 - Loss_EncY1)**2 + 1e-7

    # Calculate distance penalty term
    Loss_DistancePenalty = Loss_CenterDist / Loss_EncDiagonal

    # Calculate aspect ratio consistency term 'v'
    Loss_V = (4.0 / (np.pi ** 2)) * torch.pow(
        torch.atan(Loss_TrueBBoxes[:, 2] / (Loss_TrueBBoxes[:, 3] + 1e-7)) - 
        torch.atan(Loss_PredBBoxes[:, 2] / (Loss_PredBBoxes[:, 3] + 1e-7)), 
        2
    )
    
    # Calculate weight term 'alpha' with no gradient tracking
    with torch.no_grad():
        Loss_Alpha = Loss_V / (1.0 - Loss_IoU + Loss_V + 1e-7)
        
    # Combine terms to compute CIoU score
    Loss_CIoU = Loss_IoU - (Loss_DistancePenalty + Loss_Alpha * Loss_V)
    # Compute CIoU loss (bounded between 0 and 2)
    Loss_CIoULoss = 1.0 - Loss_CIoU
    # Return mean CIoU loss across batch
    return Loss_CIoULoss.mean()

# =============================================================================
# SECTION: Model Training Function
# =============================================================================
def train_model(Train_ModelInstance, Train_DataLoaderInstance, Train_Epochs=5):
    # Check if GPU acceleration is available and map device accordingly
    Train_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to selected device
    Train_ModelInstance.to(Train_Device)
    # Put model in active training state
    Train_ModelInstance.train()
    
    # Instantiate standard Adam optimizer for model parameters
    Train_Optimizer = torch.optim.Adam(Train_ModelInstance.parameters(), lr=0.001)
    # Initialize PyTorch GradScaler for float16 mixed precision training
    Train_Scaler = torch.cuda.amp.GradScaler(enabled=(Train_Device.type == "cuda"))

    # Loop through training epochs
    for Loop_EpochIndex in range(Train_Epochs):
        # Initialize running loss accumulator for debugging
        Train_RunningLoss = 0.0
        
        # Process batches from DataLoader
        for Loop_BatchIndex, Train_BatchData in enumerate(Train_DataLoaderInstance):
            # Extract inputs and transfer to device
            Train_Camera = Train_BatchData["camera"].to(Train_Device)
            Train_RadarRD = Train_BatchData["radar_rd"].to(Train_Device)
            Train_RadarRA = Train_BatchData["radar_ra"].to(Train_Device)
            # Extract labels and transfer to device
            Train_BBoxTrue = Train_BatchData["bbox"].to(Train_Device)
            Train_LabelTrue = Train_BatchData["label"].to(Train_Device)

            # Reset optimizer gradients
            Train_Optimizer.zero_grad()

            # Execute forward pass within autocast scope to utilize float16 acceleration
            with torch.cuda.amp.autocast(enabled=(Train_Device.type == "cuda")):
                # Generate predictions
                Train_ClassLogit, Train_BBoxPred, _ = Train_ModelInstance(
                    Train_Camera, Train_RadarRD, Train_RadarRA
                )
                
                # Compute Focal Loss for class confidence
                Train_Focal = compute_focal_loss(Train_ClassLogit, Train_LabelTrue)
                # Compute CIoU Loss for bounding box regressor
                Train_CIoU = compute_ciou_loss(Train_BBoxPred, Train_BBoxTrue)
                # Total loss with equal weighting coefficients alpha = 1.0, beta = 1.0
                Train_TotalLoss = Train_Focal + Train_CIoU

            # Run backward pass with scaler scaling to prevent underflow
            Train_Scaler.scale(Train_TotalLoss).backward()
            # Update parameters via scaler step
            Train_Scaler.step(Train_Optimizer)
            # Update scale factor for next iteration
            Train_Scaler.update()

            # Accumulate running loss value
            Train_RunningLoss += Train_TotalLoss.item()

        # Print training progress metrics per epoch
        print(f"Epoch {Loop_EpochIndex + 1}/{Train_Epochs} - Mean Loss: {Train_RunningLoss / len(Train_DataLoaderInstance):.4f}")

# =============================================================================
# SECTION: Model Evaluation Function (Ablation testing)
# =============================================================================
def evaluate_model(Eval_ModelInstance, Eval_DataLoaderInstance):
    # Check if GPU acceleration is available
    Eval_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Put model in evaluation state
    Eval_ModelInstance.eval()
    # Move model to device
    Eval_ModelInstance.to(Eval_Device)
    
    # Initialize lists to store computed IoU values
    Eval_IoUList = []

    # Disable gradient tracking for speed and memory efficiency
    with torch.no_grad():
        # Process batches from DataLoader
        for Eval_BatchData in Eval_DataLoaderInstance:
            # Transfer input variables to device
            Eval_Camera = Eval_BatchData["camera"].to(Eval_Device)
            Eval_RadarRD = Eval_BatchData["radar_rd"].to(Eval_Device)
            Eval_RadarRA = Eval_BatchData["radar_ra"].to(Eval_Device)
            Eval_BBoxTrue = Eval_BatchData["bbox"].to(Eval_Device)

            # Generate predictions
            _, Eval_BBoxPred, _ = Eval_ModelInstance(
                Eval_Camera, Eval_RadarRD, Eval_RadarRA
            )

            # Calculate IoU for each predicted box in the batch
            Eval_PredX1 = Eval_BBoxPred[:, 0] - Eval_BBoxPred[:, 2] / 2.0
            Eval_PredY1 = Eval_BBoxPred[:, 1] - Eval_BBoxPred[:, 3] / 2.0
            Eval_PredX2 = Eval_BBoxPred[:, 0] + Eval_BBoxPred[:, 2] / 2.0
            Eval_PredY2 = Eval_BBoxPred[:, 1] + Eval_BBoxPred[:, 3] / 2.0

            Eval_TrueX1 = Eval_BBoxTrue[:, 0] - Eval_BBoxTrue[:, 2] / 2.0
            Eval_TrueY1 = Eval_BBoxTrue[:, 1] - Eval_BBoxTrue[:, 3] / 2.0
            Eval_TrueX2 = Eval_BBoxTrue[:, 0] + Eval_BBoxTrue[:, 2] / 2.0
            Eval_TrueY2 = Eval_BBoxTrue[:, 1] + Eval_BBoxTrue[:, 3] / 2.0

            Eval_PredArea = (Eval_PredX2 - Eval_PredX1).clamp(min=0) * (Eval_PredY2 - Eval_PredY1).clamp(min=0)
            Eval_TrueArea = (Eval_TrueX2 - Eval_TrueX1).clamp(min=0) * (Eval_TrueY2 - Eval_TrueY1).clamp(min=0)

            Eval_InterX1 = torch.max(Eval_PredX1, Eval_TrueX1)
            Eval_InterY1 = torch.max(Eval_PredY1, Eval_TrueY1)
            Eval_InterX2 = torch.min(Eval_PredX2, Eval_TrueX2)
            Eval_InterY2 = torch.min(Eval_PredY2, Eval_TrueY2)

            Eval_InterArea = (Eval_InterX2 - Eval_InterX1).clamp(min=0) * (Eval_InterY2 - Eval_InterY1).clamp(min=0)
            Eval_UnionArea = Eval_PredArea + Eval_TrueArea - Eval_InterArea + 1e-7
            Eval_IoU = Eval_InterArea / Eval_UnionArea
            
            # Extend lists with computed batch IoU values
            Eval_IoUList.extend(Eval_IoU.cpu().numpy().tolist())

    # Return average mean IoU over whole validation set
    return np.mean(Eval_IoUList)

# =============================================================================
# SECTION: Main Script Execution for Training & Ablation Table
# =============================================================================
if __name__ == "__main__":
    # Hyperparameters
    MAIN_BATCH_SIZE = 4
    MAIN_EPOCHS = 3
    MAIN_LATENT_DIM = 256

    print("#############################################")
    print("# TASK 3: End-to-End Training & Ablation")
    print("#############################################\n")

    # Create datasets
    print("Creating Simulated Datasets...")
    Main_TrainDataset = RadVisionDataset(DatasetInit_NumSamples=32, DatasetInit_OcclusionProb=0.2)
    Main_TestDatasetClean = RadVisionDataset(DatasetInit_NumSamples=16, DatasetInit_OcclusionProb=0.0)
    Main_TestDatasetOccluded = RadVisionDataset(DatasetInit_NumSamples=16, DatasetInit_OcclusionProb=1.0)

    # Create data loaders
    Main_TrainLoader = DataLoader(Main_TrainDataset, batch_size=MAIN_BATCH_SIZE, shuffle=True)
    Main_TestLoaderClean = DataLoader(Main_TestDatasetClean, batch_size=MAIN_BATCH_SIZE, shuffle=False)
    Main_TestLoaderOccluded = DataLoader(Main_TestDatasetOccluded, batch_size=MAIN_BATCH_SIZE, shuffle=False)

    # 1. Train and Evaluate Vision-Only Branch
    print("\n---------------------------------------------")
    print("Training Vision-Only Model...")
    print("---------------------------------------------")
    Main_VisionOnlyModel = RadVisionFusionModel(ModelInit_LatentDim=MAIN_LATENT_DIM, ModelInit_Mode="vision_only")
    train_model(Main_VisionOnlyModel, Main_TrainLoader, Train_Epochs=MAIN_EPOCHS)
    Main_VisionCleanIoU = evaluate_model(Main_VisionOnlyModel, Main_TestLoaderClean)
    Main_VisionOccludedIoU = evaluate_model(Main_VisionOnlyModel, Main_TestLoaderOccluded)

    # 2. Train and Evaluate Radar-Only Branch
    print("\n---------------------------------------------")
    print("Training Radar-Only Model...")
    print("---------------------------------------------")
    Main_RadarOnlyModel = RadVisionFusionModel(ModelInit_LatentDim=MAIN_LATENT_DIM, ModelInit_Mode="radar_only")
    train_model(Main_RadarOnlyModel, Main_TrainLoader, Train_Epochs=MAIN_EPOCHS)
    Main_RadarCleanIoU = evaluate_model(Main_RadarOnlyModel, Main_TestLoaderClean)
    Main_RadarOccludedIoU = evaluate_model(Main_RadarOnlyModel, Main_TestLoaderOccluded)

    # 3. Train and Evaluate Full RadVision Fused Model
    print("\n---------------------------------------------")
    print("Training RadVision (Fused) Model...")
    print("---------------------------------------------")
    Main_FusedModel = RadVisionFusionModel(ModelInit_LatentDim=MAIN_LATENT_DIM, ModelInit_Mode="fused")
    train_model(Main_FusedModel, Main_TrainLoader, Train_Epochs=MAIN_EPOCHS)
    Main_FusedCleanIoU = evaluate_model(Main_FusedModel, Main_TestLoaderClean)
    Main_FusedOccludedIoU = evaluate_model(Main_FusedModel, Main_TestLoaderOccluded)

    # Print Ablation Table Results
    print("\n=============================================")
    print("[ABLATION] Modality Comparison Table")
    print("=============================================")
    print(f"{'Model':<18} | {'Camera':<8} | {'Radar':<8} | {'Test Clean IoU':<14} | {'Test Occluded IoU':<17}")
    print("-" * 75)
    print(f"{'Vision-Only':<18} | {'Active':<8} | {'Ignored':<8} | {Main_VisionCleanIoU:<14.4f} | {Main_VisionOccludedIoU:<17.4f}")
    print(f"{'Radar-Only':<18} | {'Ignored':<8} | {'Active':<8} | {Main_RadarCleanIoU:<14.4f} | {Main_RadarOccludedIoU:<17.4f}")
    print(f"{'RadVision (Fused)':<18} | {'Active':<8} | {'Active':<8} | {Main_FusedCleanIoU:<14.4f} | {Main_FusedOccludedIoU:<17.4f}")
    print("=============================================\n")
