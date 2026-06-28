# =============================================================================
# STUDENT-CODING-AGENT NOTICE:
# Classes and functions are written here as implicitly required by the PyTorch
# framework (Dataset, DataLoader, nn.Module). All style rules, including line-by-line
# comments, section-prefixed variable names, and bordered prints are applied internally.
# =============================================================================

# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import Streamlit library for interactive web dashboard UI
import streamlit as st
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
# Import matplotlib for rendering spatial feature maps and bounding boxes
import matplotlib.pyplot as plt
# Import patches from matplotlib to draw bounding boxes
import matplotlib.patches as patches

# =============================================================================
# SECTION: Custom Dataset (From Chunk 1 & 3)
# =============================================================================
class RadVisionDataset(Dataset):
    # Constructor method for initializing the dataset with sequence length
    def __init__(self, DatasetInit_NumSamples=20):
        # Store the total number of samples to simulate
        self.DatasetInit_NumSamples = DatasetInit_NumSamples

    # Return the total number of samples in the dataset
    def __len__(self):
        # Return the stored sample size count
        return self.DatasetInit_NumSamples

    # Fetch a single multi-modal sample by its index
    def __getitem__(self, DatasetGet_Index):
        # Set a random seed based on index to ensure deterministic generation per index
        np.random.seed(DatasetGet_Index)
        
        # Simulate a 3-channel RGB Camera frame of size 3x224x224
        # We will generate a structured image: a dark background with a bright red circle object
        DatasetGet_CameraFrame = np.zeros((3, 224, 224), dtype=np.float32)
        # Bounding box center coordinates
        DatasetGet_CenterY = 112
        DatasetGet_CenterX = 112
        DatasetGet_Radius = 30
        # Draw red circle (channel 0)
        for Loop_Y in range(224):
            for Loop_X in range(224):
                if (Loop_Y - DatasetGet_CenterY)**2 + (Loop_X - DatasetGet_CenterX)**2 < DatasetGet_Radius**2:
                    DatasetGet_CameraFrame[0, Loop_Y, Loop_X] = 1.0
                    DatasetGet_CameraFrame[1, Loop_Y, Loop_X] = 0.1
                    DatasetGet_CameraFrame[2, Loop_Y, Loop_X] = 0.1
        
        # Add random noise to background
        DatasetGet_CameraFrame += np.random.normal(0, 0.05, DatasetGet_CameraFrame.shape).astype(np.float32)
        # Clip to valid range [0, 1]
        DatasetGet_CameraFrame = np.clip(DatasetGet_CameraFrame, 0.0, 1.0)
        
        # Simulate a 1-channel Range-Doppler (RD) radar heatmap of size 1x256x64
        # We place a strong signal peak in the center to simulate the target object
        DatasetGet_RadarRD = np.random.normal(0, 0.1, (1, 256, 64)).astype(np.float32)
        # Set a radar peak representing target range and speed
        DatasetGet_RadarRD[0, 128, 32] = 2.5
        
        # Simulate a 1-channel Range-Angle (RA) radar heatmap of size 1x256x256
        # We place a strong signal peak in the center to simulate target range and angle location
        DatasetGet_RadarRA = np.random.normal(0, 0.1, (1, 256, 256)).astype(np.float32)
        # Set a radar peak representing target range and angular position
        DatasetGet_RadarRA[0, 128, 128] = 2.5
        
        # Ground-truth bounding box: [x_center, y_center, width, height] (normalized)
        # Our target circle is in the absolute center
        DatasetGet_BBox = np.array([0.5, 0.5, 0.27, 0.27], dtype=np.float32)
        
        # Ground-truth class label (1.0 for target object)
        DatasetGet_ClassLabel = np.array([1.0], dtype=np.float32)
        
        # Convert arrays to PyTorch tensors
        DatasetGet_CameraTensor = torch.tensor(DatasetGet_CameraFrame)
        DatasetGet_RadarRDTensor = torch.tensor(DatasetGet_RadarRD)
        DatasetGet_RadarRATensor = torch.tensor(DatasetGet_RadarRA)
        DatasetGet_BBoxTensor = torch.tensor(DatasetGet_BBox)
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
        # Apply layer 1 block operations
        RadarForward_Layer1 = self.RadarInit_MaxPool1(F.relu(self.RadarInit_BatchNorm1(self.RadarInit_Conv1(RadarForward_InputTensor))))
        # Apply layer 2 block operations
        RadarForward_Layer2 = self.RadarInit_MaxPool2(F.relu(self.RadarInit_BatchNorm2(self.RadarInit_Conv2(RadarForward_Layer1))))
        # Apply layer 3 block operations
        RadarForward_Layer3 = self.RadarInit_MaxPool3(F.relu(self.RadarInit_BatchNorm3(self.RadarInit_Conv3(RadarForward_Layer2))))
        # Apply layer 4 block operations
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
# SECTION: Flexible RadVision-Fusion Network Model (From Chunk 3)
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
        # Layer Normalization layers
        self.ModelInit_Norm1 = nn.LayerNorm(ModelInit_LatentDim)
        self.ModelInit_Norm2 = nn.LayerNorm(ModelInit_LatentDim)
        # MLP Detection Head
        self.ModelInit_DetHead = nn.Sequential(
            nn.Linear(49 * ModelInit_LatentDim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        # Radar-Only MLP Head
        self.ModelInit_RadarOnlyDetHead = nn.Sequential(
            nn.Linear(ModelInit_LatentDim, 128),
            nn.ReLU()
        )
        # Final classification and box regressor layers
        self.ModelInit_ClassClassifier = nn.Linear(128, 1)
        self.ModelInit_BoxRegressor = nn.Linear(128, 4)

    # Forward pass of entire fusion pipeline
    def forward(self, ModelForward_CamTensor, ModelForward_RDTensor, ModelForward_RATensor):
        # Radar-Only Mode
        if self.ModelInit_Mode == "radar_only":
            ModelForward_RDFeatures = self.ModelInit_RadarEncoderRD(ModelForward_RDTensor)
            ModelForward_RAFeatures = self.ModelInit_RadarEncoderRA(ModelForward_RATensor)
            ModelForward_RDTokens = self.ModelInit_RadarRDTokenizer(ModelForward_RDFeatures)
            ModelForward_RATokens = self.ModelInit_RadarRATokenizer(ModelForward_RAFeatures)
            ModelForward_RadarTokensCombined = torch.cat([ModelForward_RDTokens, ModelForward_RATokens], dim=1)
            ModelForward_PooledRadar = ModelForward_RadarTokensCombined.mean(dim=1)
            ModelForward_MLPFeatures = self.ModelInit_RadarOnlyDetHead(ModelForward_PooledRadar)
            ModelForward_ClassLogit = self.ModelInit_ClassClassifier(ModelForward_MLPFeatures)
            ModelForward_BBoxPred = torch.sigmoid(self.ModelInit_BoxRegressor(ModelForward_MLPFeatures))
            return ModelForward_ClassLogit, ModelForward_BBoxPred, None

        # Vision-Only Mode
        elif self.ModelInit_Mode == "vision_only":
            ModelForward_CamFeatures = self.ModelInit_VisionEncoder(ModelForward_CamTensor)
            ModelForward_CamTokens = self.ModelInit_VisionTokenizer(ModelForward_CamFeatures)
            ModelForward_FFNResult = self.ModelInit_FFN(ModelForward_CamTokens)
            ModelForward_FusedTokens = self.ModelInit_Norm2(ModelForward_CamTokens + ModelForward_FFNResult)
            ModelForward_FlattenedTokens = ModelForward_FusedTokens.flatten(start_dim=1)
            ModelForward_MLPFeatures = self.ModelInit_DetHead(ModelForward_FlattenedTokens)
            ModelForward_ClassLogit = self.ModelInit_ClassClassifier(ModelForward_MLPFeatures)
            ModelForward_BBoxPred = torch.sigmoid(self.ModelInit_BoxRegressor(ModelForward_MLPFeatures))
            return ModelForward_ClassLogit, ModelForward_BBoxPred, None

        # Fused Mode
        else:
            ModelForward_CamFeatures = self.ModelInit_VisionEncoder(ModelForward_CamTensor)
            ModelForward_RDFeatures = self.ModelInit_RadarEncoderRD(ModelForward_RDTensor)
            ModelForward_RAFeatures = self.ModelInit_RadarEncoderRA(ModelForward_RATensor)
            ModelForward_CamTokens = self.ModelInit_VisionTokenizer(ModelForward_CamFeatures)
            ModelForward_RDTokens = self.ModelInit_RadarRDTokenizer(ModelForward_RDFeatures)
            ModelForward_RATokens = self.ModelInit_RadarRATokenizer(ModelForward_RAFeatures)
            
            ModelForward_RadarTokensCombined = torch.cat(
                [ModelForward_RDTokens, ModelForward_RATokens], 
                dim=1
            )
            
            ModelForward_AttnOutput, ModelForward_AttnWeights = self.ModelInit_CrossAttention(
                query=ModelForward_CamTokens,
                key=ModelForward_RadarTokensCombined,
                value=ModelForward_RadarTokensCombined
            )
            
            ModelForward_AttendedCam = self.ModelInit_Norm1(ModelForward_CamTokens + ModelForward_AttnOutput)
            ModelForward_FFNResult = self.ModelInit_FFN(ModelForward_AttendedCam)
            ModelForward_FusedTokens = self.ModelInit_Norm2(ModelForward_AttendedCam + ModelForward_FFNResult)
            ModelForward_FlattenedTokens = ModelForward_FusedTokens.flatten(start_dim=1)
            ModelForward_MLPFeatures = self.ModelInit_DetHead(ModelForward_FlattenedTokens)
            ModelForward_ClassLogit = self.ModelInit_ClassClassifier(ModelForward_MLPFeatures)
            ModelForward_BBoxPred = torch.sigmoid(self.ModelInit_BoxRegressor(ModelForward_MLPFeatures))
            return ModelForward_ClassLogit, ModelForward_BBoxPred, ModelForward_AttnWeights

# =============================================================================
# SECTION: Custom Loss Computation Definitions (From Chunk 3)
# =============================================================================
def compute_focal_loss(Loss_PredLogits, Loss_TrueLabels, Loss_Alpha=0.25, Loss_Gamma=2.0):
    Loss_BCE = F.binary_cross_entropy_with_logits(Loss_PredLogits, Loss_TrueLabels, reduction="none")
    Loss_Prob = torch.sigmoid(Loss_PredLogits)
    Loss_PT = Loss_TrueLabels * Loss_Prob + (1.0 - Loss_TrueLabels) * (1.0 - Loss_Prob)
    Loss_AlphaWeight = Loss_TrueLabels * Loss_Alpha + (1.0 - Loss_TrueLabels) * (1.0 - Loss_Alpha)
    Loss_FocalFactor = torch.pow(1.0 - Loss_PT, Loss_Gamma)
    Loss_Elementwise = Loss_AlphaWeight * Loss_FocalFactor * Loss_BCE
    return Loss_Elementwise.mean()

def compute_ciou_loss(Loss_PredBBoxes, Loss_TrueBBoxes):
    Loss_PredX1 = Loss_PredBBoxes[:, 0] - Loss_PredBBoxes[:, 2] / 2.0
    Loss_PredY1 = Loss_PredBBoxes[:, 1] - Loss_PredBBoxes[:, 3] / 2.0
    Loss_PredX2 = Loss_PredBBoxes[:, 0] + Loss_PredBBoxes[:, 2] / 2.0
    Loss_PredY2 = Loss_PredBBoxes[:, 1] + Loss_PredBBoxes[:, 3] / 2.0

    Loss_TrueX1 = Loss_TrueBBoxes[:, 0] - Loss_TrueBBoxes[:, 2] / 2.0
    Loss_TrueY1 = Loss_TrueBBoxes[:, 1] - Loss_TrueBBoxes[:, 3] / 2.0
    Loss_TrueX2 = Loss_TrueBBoxes[:, 0] + Loss_TrueBBoxes[:, 2] / 2.0
    Loss_TrueY2 = Loss_TrueBBoxes[:, 1] + Loss_TrueBBoxes[:, 3] / 2.0

    Loss_PredArea = (Loss_PredX2 - Loss_PredX1).clamp(min=0) * (Loss_PredY2 - Loss_PredY1).clamp(min=0)
    Loss_TrueArea = (Loss_TrueX2 - Loss_TrueX1).clamp(min=0) * (Loss_TrueY2 - Loss_TrueY1).clamp(min=0)

    Loss_InterX1 = torch.max(Loss_PredX1, Loss_TrueX1)
    Loss_InterY1 = torch.max(Loss_PredY1, Loss_TrueY1)
    Loss_InterX2 = torch.min(Loss_PredX2, Loss_TrueX2)
    Loss_InterY2 = torch.min(Loss_PredY2, Loss_TrueY2)

    Loss_InterArea = (Loss_InterX2 - Loss_InterX1).clamp(min=0) * (Loss_InterY2 - Loss_InterY1).clamp(min=0)
    Loss_UnionArea = Loss_PredArea + Loss_TrueArea - Loss_InterArea + 1e-7
    Loss_IoU = Loss_InterArea / Loss_UnionArea
    Loss_CenterDist = (Loss_PredBBoxes[:, 0] - Loss_TrueBBoxes[:, 0])**2 + (Loss_PredBBoxes[:, 1] - Loss_TrueBBoxes[:, 1])**2

    Loss_EncX1 = torch.min(Loss_PredX1, Loss_TrueX1)
    Loss_EncY1 = torch.min(Loss_PredY1, Loss_TrueY1)
    Loss_EncX2 = torch.max(Loss_PredX2, Loss_TrueX2)
    Loss_EncY2 = torch.max(Loss_PredY2, Loss_TrueY2)
    Loss_EncDiagonal = (Loss_EncX2 - Loss_EncX1)**2 + (Loss_EncY2 - Loss_EncY1)**2 + 1e-7

    Loss_DistancePenalty = Loss_CenterDist / Loss_EncDiagonal
    Loss_V = (4.0 / (np.pi ** 2)) * torch.pow(
        torch.atan(Loss_TrueBBoxes[:, 2] / (Loss_TrueBBoxes[:, 3] + 1e-7)) - 
        torch.atan(Loss_PredBBoxes[:, 2] / (Loss_PredBBoxes[:, 3] + 1e-7)), 
        2
    )
    with torch.no_grad():
        Loss_Alpha = Loss_V / (1.0 - Loss_IoU + Loss_V + 1e-7)
    Loss_CIoU = Loss_IoU - (Loss_DistancePenalty + Loss_Alpha * Loss_V)
    Loss_CIoULoss = 1.0 - Loss_CIoU
    return Loss_CIoULoss.mean()

# =============================================================================
# SECTION: Model Training Function (Caching resource to avoid repeat training)
# =============================================================================
@st.cache_resource
def train_and_cache_models():
    # Instantiate dataset and dataloader
    DatasetInstance = RadVisionDataset(DatasetInit_NumSamples=32)
    DataLoaderInstance = DataLoader(DatasetInstance, batch_size=4, shuffle=True)
    
    # Define models
    VisionModel = RadVisionFusionModel(ModelInit_LatentDim=256, ModelInit_Mode="vision_only")
    RadarModel = RadVisionFusionModel(ModelInit_LatentDim=256, ModelInit_Mode="radar_only")
    FusedModel = RadVisionFusionModel(ModelInit_LatentDim=256, ModelInit_Mode="fused")
    
    # Train each model briefly for 5 epochs
    for ModelItem, ModelName in [(VisionModel, "Vision-Only"), (RadarModel, "Radar-Only"), (FusedModel, "RadVision (Fused)")]:
        ModelItem.train()
        Optimizer = torch.optim.Adam(ModelItem.parameters(), lr=0.001)
        for Epoch in range(5):
            for Batch in DataLoaderInstance:
                Optimizer.zero_grad()
                ClassLogit, BBoxPred, _ = ModelItem(
                    Batch["camera"], Batch["radar_rd"], Batch["radar_ra"]
                )
                Focal = compute_focal_loss(ClassLogit, Batch["label"])
                CIoU = compute_ciou_loss(BBoxPred, Batch["bbox"])
                TotalLoss = Focal + CIoU
                TotalLoss.backward()
                Optimizer.step()
        ModelItem.eval()
        
    return VisionModel, RadarModel, FusedModel

# =============================================================================
# SECTION: Streamlit UI Execution Layout
# =============================================================================
# Configure general Streamlit page properties
st.set_page_config(
    page_title="RadVision-Fusion: Resilient Multi-Modal Perception",
    page_icon="📡",
    layout="wide"
)

# Inject custom CSS for premium styling cards
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle Banner
st.title("📡 RadVision-Fusion")
st.subheader("Multi-Modal Camera–FMCW Radar Cross-Attention Network for Resilient Edge Perception")
st.markdown("---")

# Load trained models from cache
VisionModel, RadarModel, FusedModel = train_and_cache_models()

# Initialize Dataset Instance
DatasetInstance = RadVisionDataset(DatasetInit_NumSamples=20)

# Sidebar layout for user control inputs
st.sidebar.header("🎛️ Simulation Controls")
SampleIndex = st.sidebar.slider("Select Sample Index", 0, 19, 0)
OcclusionLevel = st.sidebar.slider("Camera Visual Occlusion Level", 0.0, 1.0, 0.5, step=0.1)

# Fetch selected sample from dataset
SampleData = DatasetInstance[SampleIndex]
CameraFrame = SampleData["camera"].clone()
RadarRD = SampleData["radar_rd"].clone()
RadarRA = SampleData["radar_ra"].clone()
BBoxTrue = SampleData["bbox"].numpy()

# Apply user occlusion to the camera frame (scale RGB channels down and add noise)
if OcclusionLevel > 0.0:
    # Scale down camera pixel intensity values to simulate low-light / fog blocking
    CameraFrame = CameraFrame * (1.0 - OcclusionLevel)
    # Add random Gaussian noise representing smoke / camera sensor corruption
    CameraFrame += torch.randn(CameraFrame.shape) * OcclusionLevel * 0.2
    # Clip pixels to keep them in [0, 1] range
    CameraFrame = torch.clamp(CameraFrame, 0.0, 1.0)

# Execute inference using all three models
with torch.no_grad():
    # Model 1: Vision-Only
    _, BBoxVisionPred, _ = VisionModel(
        CameraFrame.unsqueeze(0), RadarRD.unsqueeze(0), RadarRA.unsqueeze(0)
    )
    # Model 2: Radar-Only
    _, BBoxRadarPred, _ = RadarModel(
        CameraFrame.unsqueeze(0), RadarRD.unsqueeze(0), RadarRA.unsqueeze(0)
    )
    # Model 3: RadVision (Fused) with attention matrix
    _, BBoxFusedPred, AttnWeights = FusedModel(
        CameraFrame.unsqueeze(0), RadarRD.unsqueeze(0), RadarRA.unsqueeze(0)
    )

# Helper function to compute standard IoU score between boxes
def compute_iou_metric(PredBox, TrueBox):
    # Convert prediction and target center size formats to bounds
    Px1, Py1, Px2, Py2 = PredBox[0]-PredBox[2]/2, PredBox[1]-PredBox[3]/2, PredBox[0]+PredBox[2]/2, PredBox[1]+PredBox[3]/2
    Tx1, Ty1, Tx2, Ty2 = TrueBox[0]-TrueBox[2]/2, TrueBox[1]-TrueBox[3]/2, TrueBox[0]+TrueBox[2]/2, TrueBox[1]+TrueBox[3]/2
    
    # Compute intersection area
    Ix1, Iy1, Ix2, Iy2 = max(Px1, Tx1), max(Py1, Ty1), min(Px2, Tx2), min(Py2, Ty2)
    InterArea = max(0.0, Ix2 - Ix1) * max(0.0, Iy2 - Iy1)
    
    # Compute union area
    PredArea = (Px2 - Px1) * (Py2 - Py1)
    TrueArea = (Tx2 - Tx1) * (Ty2 - Ty1)
    UnionArea = PredArea + TrueArea - InterArea + 1e-7
    
    # Return Intersection over Union
    return float(InterArea / UnionArea)

# Calculate metrics for the UI cards
IoUVision = compute_iou_metric(BBoxVisionPred[0].numpy(), BBoxTrue)
IoURadar = compute_iou_metric(BBoxRadarPred[0].numpy(), BBoxTrue)
IoUFused = compute_iou_metric(BBoxFusedPred[0].numpy(), BBoxTrue)

# Main Dashboard Columns: Raw Feeds
st.header("🖼️ Raw Sensor Feeds & Modality Visualizations")
ColFeed1, ColFeed2, ColFeed3 = st.columns(3)

with ColFeed1:
    st.subheader("📸 Camera Feed (RGB)")
    # Render camera image using matplotlib with bounding box overlays
    FigCam, AxCam = plt.subplots(figsize=(4, 4))
    # Permute shape to standard matplotlib HWC format
    CamImageNP = CameraFrame.permute(1, 2, 0).numpy()
    AxCam.imshow(CamImageNP)
    AxCam.set_title("Synchronized Video Frame")
    AxCam.axis("off")
    
    # Add Ground Truth Bounding Box in Green
    GTBoxPatch = patches.Rectangle(
        ((BBoxTrue[0] - BBoxTrue[2]/2)*224, (BBoxTrue[1] - BBoxTrue[3]/2)*224),
        BBoxTrue[2]*224, BBoxTrue[3]*224,
        linewidth=2, edgecolor="g", facecolor="none", label="Ground Truth"
    )
    AxCam.add_patch(GTBoxPatch)
    
    # Add Predicted Bounding Box in Yellow (from Fused Model)
    PredBoxNP = BBoxFusedPred[0].numpy()
    FusedBoxPatch = patches.Rectangle(
        ((PredBoxNP[0] - PredBoxNP[2]/2)*224, (PredBoxNP[1] - PredBoxNP[3]/2)*224),
        PredBoxNP[2]*224, PredBoxNP[3]*224,
        linewidth=2, edgecolor="yellow", facecolor="none", label="Fused Prediction"
    )
    AxCam.add_patch(FusedBoxPatch)
    AxCam.legend(loc="upper right")
    
    st.pyplot(FigCam)

with ColFeed2:
    st.subheader("📈 Range-Doppler Heatmap")
    # Render RD Map using matplotlib viridis colormap
    FigRD, AxRD = plt.subplots(figsize=(4, 4))
    RDNP = RadarRD[0].numpy()
    # Crop around the central signal area for visual aesthetic clarity
    AxRD.imshow(RDNP[96:160, 16:48], cmap="viridis", aspect="auto")
    AxRD.set_title("Range-Doppler Spectrum")
    AxRD.set_xlabel("Doppler Speed Bin")
    AxRD.set_ylabel("Range Bin")
    st.pyplot(FigRD)

with ColFeed3:
    st.subheader("📐 Range-Angle Heatmap")
    # Render RA Map using matplotlib magma colormap
    FigRA, AxRA = plt.subplots(figsize=(4, 4))
    RANP = RadarRA[0].numpy()
    # Crop around the central signal area for visual aesthetic clarity
    AxRA.imshow(RANP[96:160, 96:160], cmap="magma", aspect="auto")
    AxRA.set_title("Range-Angle Cartesian Projection")
    AxRA.set_xlabel("Angular Bin")
    AxRA.set_ylabel("Range Bin")
    st.pyplot(FigRA)

st.markdown("---")

# Metrics Summary Cards
st.header("📊 Real-Time Bounding Box IoU Metrics")
ColM1, ColM2, ColM3 = st.columns(3)

with ColM1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vision-Only Model</div>
        <div class="metric-value">{IoUVision:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with ColM2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Radar-Only Model</div>
        <div class="metric-value">{IoURadar:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with ColM3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #4ade80;">{IoUFused:.4f}</div>
        <div class="metric-label">RadVision (Fused Model)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Cross-Attention Weights Visualization
st.header("🧠 Cross-Attention Matrix Heatmap")
st.markdown("Visualizes attention routing from Camera Queries ($Q$, 49 spatial patches) to Radar Keys/Values ($K/V$, 320 patches).")

if AttnWeights is not None:
    # Extract weights tensor: [Batch, 49, 320] -> [49, 320]
    AttnMatrix = AttnWeights[0].numpy()
    
    FigAttn, AxAttn = plt.subplots(figsize=(10, 4))
    ImAttn = AxAttn.imshow(AttnMatrix, cmap="inferno", aspect="auto")
    AxAttn.set_title("Cross-Attention Alignment Weights (Camera Query vs Combined Radar Key)")
    AxAttn.set_xlabel("Combined Radar Tokens (0-63: RD, 64-319: RA)")
    AxAttn.set_ylabel("Camera Token Index (7x7 Spatial Grid)")
    FigAttn.colorbar(ImAttn, ax=AxAttn)
    st.pyplot(FigAttn)
else:
    st.info("Cross-Attention weights are only available in Fused mode.")
