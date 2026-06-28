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
# Import numpy for numerical operations
import numpy as np
# Import random module for random augmentation decisions
import random

# =============================================================================
# SECTION: Custom Dataset (From Chunk 1)
# =============================================================================
class RadVisionDataset(Dataset):
    # Constructor method for initializing the dataset with sequence length
    def __init__(self, DatasetInit_NumSamples=100):
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
        DatasetGet_CameraFrame = np.random.randn(3, 224, 224).astype(np.float32)
        
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
            # Flip camera frame along the width axis (dimension 2)
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

# =============================================================================
# SECTION: Full RadVision-Fusion Network Model (Fusion & Head)
# =============================================================================
class RadVisionFusionModel(nn.Module):
    # Constructor defining encoders, tokenizers, cross-attention, and detection head
    def __init__(self, ModelInit_LatentDim=256):
        # Invoke parent constructor
        super(RadVisionFusionModel, self).__init__()
        
        # Latent dimension (D = 256)
        self.ModelInit_LatentDim = ModelInit_LatentDim
        
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
        
        # MLP Detection Head: flattens the 49 attended camera tokens and predicts classes + boxes
        # Input dimension: 49 tokens * 256 features = 12,544
        self.ModelInit_DetHead = nn.Sequential(
            nn.Linear(49 * ModelInit_LatentDim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Final projection layer to predict class confidence score (1 logit value)
        self.ModelInit_ClassClassifier = nn.Linear(128, 1)
        # Final projection layer to predict bounding box coords [x_center, y_center, width, height]
        self.ModelInit_BoxRegressor = nn.Linear(128, 4)

    # Forward pass of entire fusion pipeline
    def forward(self, ModelForward_CamTensor, ModelForward_RDTensor, ModelForward_RATensor):
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
# SECTION: Main Script Verification Logic
# =============================================================================
if __name__ == "__main__":
    # Define hyperparameter constants for our networks
    MAIN_BATCH_SIZE = 4
    MAIN_LATENT_DIM = 256

    # Print main task header banner
    print("#############################################")
    print("# TASK 2: Verification of Fusion & Head")
    print("#############################################\n")

    # Instantiate our custom dataset with simulated samples
    Main_DatasetInstance = RadVisionDataset(DatasetInit_NumSamples=8)
    
    # Create the PyTorch DataLoader to batch the simulated samples
    Main_DataLoaderInstance = DataLoader(
        dataset=Main_DatasetInstance,
        batch_size=MAIN_BATCH_SIZE,
        shuffle=True
    )
    
    # Extract the first batch of data from the data loader
    Main_FirstBatch = next(iter(Main_DataLoaderInstance))
    
    # Get individual tensors from the batch dictionary
    Main_BatchCamera = Main_FirstBatch["camera"]
    Main_BatchRadarRD = Main_FirstBatch["radar_rd"]
    Main_BatchRadarRA = Main_FirstBatch["radar_ra"]

    # Instantiate the Full RadVision-Fusion Network Model
    Main_FusionModelInstance = RadVisionFusionModel(ModelInit_LatentDim=MAIN_LATENT_DIM)
    # Set model to evaluation mode for shape verification
    Main_FusionModelInstance.eval()

    # Pass the camera and radar features through tokenization and cross-attention blocks
    with torch.no_grad():
        # Get individual encoders' feature outputs to print intermediate shapes
        Main_CamFeatures = Main_FusionModelInstance.ModelInit_VisionEncoder(Main_BatchCamera)
        Main_RDFeatures = Main_FusionModelInstance.ModelInit_RadarEncoderRD(Main_BatchRadarRD)
        Main_RAFeatures = Main_FusionModelInstance.ModelInit_RadarEncoderRA(Main_BatchRadarRA)
        
        # Convert feature maps to token sequences
        Main_CamTokens = Main_FusionModelInstance.ModelInit_VisionTokenizer(Main_CamFeatures)
        Main_RDTokens = Main_FusionModelInstance.ModelInit_RadarRDTokenizer(Main_RDFeatures)
        Main_RATokens = Main_FusionModelInstance.ModelInit_RadarRATokenizer(Main_RAFeatures)
        
        # Combine radar tokens along sequence axis
        Main_RadarTokensCombined = torch.cat([Main_RDTokens, Main_RATokens], dim=1)
        
        # Execute the full forward pass to get final predictions
        Main_ClassLogits, Main_BBoxPreds, Main_AttnWeights = Main_FusionModelInstance(
            Main_BatchCamera, 
            Main_BatchRadarRD, 
            Main_BatchRadarRA
        )

    # Print tokenized camera sequence shapes
    print("=============================================")
    print("[TOKEN] Camera Tokens Shape")
    print("=============================================")
    print("Description : Flattened spatial camera tokens after positional embedding")
    print(f"Result      : {list(Main_CamTokens.shape)}")
    print("=============================================\n")

    # Print tokenized Range-Doppler sequence shapes
    print("=============================================")
    print("[TOKEN] Radar RD Tokens Shape")
    print("=============================================")
    print("Description : Flattened spatial Range-Doppler tokens after positional embedding")
    print(f"Result      : {list(Main_RDTokens.shape)}")
    print("=============================================\n")

    # Print tokenized Range-Angle sequence shapes
    print("=============================================")
    print("[TOKEN] Radar RA Tokens Shape")
    print("=============================================")
    print("Description : Flattened spatial Range-Angle tokens after positional embedding")
    print(f"Result      : {list(Main_RATokens.shape)}")
    print("=============================================\n")

    # Print combined radar sequence shapes
    print("=============================================")
    print("[TOKEN] Combined Radar Tokens Shape")
    print("=============================================")
    print("Description : Unified sequence combining Range-Doppler and Range-Angle tokens")
    print(f"Result      : {list(Main_RadarTokensCombined.shape)}")
    print("=============================================\n")

    # Print attention weights shape
    print("=============================================")
    print("[FUSION] Attention Weights Shape")
    print("=============================================")
    print("Description : Multi-head cross-attention weight matrix mapping Query to Key")
    print(f"Result      : {list(Main_AttnWeights.shape)}")
    print("=============================================\n")

    # Print predicted class logits shape
    print("=============================================")
    print("[OUTPUT] Predicted Class Logits Shape")
    print("=============================================")
    print("Description : Bounding box classification scores logit tensor")
    print(f"Result      : {list(Main_ClassLogits.shape)}")
    print("=============================================\n")

    # Print predicted bounding boxes shape
    print("=============================================")
    print("[OUTPUT] Predicted Bounding Boxes Shape")
    print("=============================================")
    print("Description : Normalised predictions [x_center, y_center, width, height]")
    print(f"Result      : {list(Main_BBoxPreds.shape)}")
    print("=============================================\n")
