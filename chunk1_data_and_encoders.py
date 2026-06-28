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
# Import numpy for numerical and array operations
import numpy as np
# Import random module for random augmentation decisions
import random

# =============================================================================
# SECTION: Custom Dataset for Multi-Modal Radar-Vision Data
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
# SECTION: Vision Encoder Branch (Ec)
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
# SECTION: Radar Encoder Branch (Er)
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
# SECTION: Main Script Verification Logic
# =============================================================================
if __name__ == "__main__":
    # Define hyperparameter constants for our networks
    MAIN_BATCH_SIZE = 4
    MAIN_LATENT_DIM = 256

    # Print main task header banner
    print("#############################################")
    print("# TASK 1: Verification of Data & Encoders")
    print("#############################################\n")

    # Instantiate our custom dataset with simulated samples
    Main_DatasetInstance = RadVisionDataset(DatasetInit_NumSamples=16)
    
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
    Main_BatchBBox = Main_FirstBatch["bbox"]
    Main_BatchLabel = Main_FirstBatch["label"]

    # Print input camera batch shapes
    print("=============================================")
    print("[INPUT] Camera Batch Shape")
    print("=============================================")
    print("Description : Tensor shape of batch of camera RGB images")
    print(f"Result      : {list(Main_BatchCamera.shape)}")
    print("=============================================\n")

    # Print input Range-Doppler batch shapes
    print("=============================================")
    print("[INPUT] Radar RD Heatmap Batch Shape")
    print("=============================================")
    print("Description : Tensor shape of batch of radar Range-Doppler heatmaps")
    print(f"Result      : {list(Main_BatchRadarRD.shape)}")
    print("=============================================\n")

    # Print input Range-Angle batch shapes
    print("=============================================")
    print("[INPUT] Radar RA Heatmap Batch Shape")
    print("=============================================")
    print("Description : Tensor shape of batch of radar Range-Angle heatmaps")
    print(f"Result      : {list(Main_BatchRadarRA.shape)}")
    print("=============================================\n")

    # Instantiate the Vision Encoder module
    Main_VisionEncoderInstance = VisionEncoder(VisionInit_LatentDim=MAIN_LATENT_DIM)
    
    # Instantiate the Radar Encoder module for Range-Doppler (RD) maps
    Main_RadarEncoderRDInstance = RadarEncoder(RadarInit_LatentDim=MAIN_LATENT_DIM)
    
    # Instantiate the Radar Encoder module for Range-Angle (RA) maps
    Main_RadarEncoderRAInstance = RadarEncoder(RadarInit_LatentDim=MAIN_LATENT_DIM)

    # Put encoders in evaluation mode to bypass batch norm tracking for verification
    Main_VisionEncoderInstance.eval()
    Main_RadarEncoderRDInstance.eval()
    Main_RadarEncoderRAInstance.eval()

    # Pass the camera batch through the vision encoder
    with torch.no_grad():
        Main_VisionOutput = Main_VisionEncoderInstance(Main_BatchCamera)
        
    # Pass the Range-Doppler batch through the corresponding radar encoder
    with torch.no_grad():
        Main_RadarRDOutput = Main_RadarEncoderRDInstance(Main_BatchRadarRD)
        
    # Pass the Range-Angle batch through the corresponding radar encoder
    with torch.no_grad():
        Main_RadarRAOutput = Main_RadarEncoderRAInstance(Main_BatchRadarRA)

    # Print vision encoder output features shape
    print("=============================================")
    print("[OUTPUT] Vision Encoder Projected Features Shape")
    print("=============================================")
    print("Description : Tensor shape of processed and projected camera features")
    print(f"Result      : {list(Main_VisionOutput.shape)}")
    print("=============================================\n")

    # Print radar Range-Doppler encoder output features shape
    print("=============================================")
    print("[OUTPUT] Radar RD Encoder Projected Features Shape")
    print("=============================================")
    print("Description : Tensor shape of processed and projected Range-Doppler features")
    print(f"Result      : {list(Main_RadarRDOutput.shape)}")
    print("=============================================\n")

    # Print radar Range-Angle encoder output features shape
    print("=============================================")
    print("[OUTPUT] Radar RA Encoder Projected Features Shape")
    print("=============================================")
    print("Description : Tensor shape of processed and projected Range-Angle features")
    print(f"Result      : {list(Main_RadarRAOutput.shape)}")
    print("=============================================\n")
