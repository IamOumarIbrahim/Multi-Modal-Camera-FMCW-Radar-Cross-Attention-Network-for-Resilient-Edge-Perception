
# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import neural network modules from PyTorch
import torch.nn as nn
# Import functional operations from PyTorch neural network tools
import torch.nn.functional as F

# =============================================================================
# SECTION: Radar Encoder Branch (Er)
# =============================================================================
class RadarEncoder(nn.Module):
    # Constructor for initializing custom CNN layers for radar processing
    def __init__(self, RadarInit_LatentDim=256):
        # Invoke the parent nn.Module constructor
        super(RadarEncoder, self).__init__()
        
        # Block 1: Conv 1x256xW -> Conv 16x128x(W/2)
        self.RadarInit_Conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm1 = nn.BatchNorm2d(16)
        self.RadarInit_MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Conv 16x128x(W/2) -> Conv 64x64x(W/4)
        self.RadarInit_Conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm2 = nn.BatchNorm2d(64)
        self.RadarInit_MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Conv 64x64x(W/4) -> Conv 128x32x(W/8)
        self.RadarInit_Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm3 = nn.BatchNorm2d(128)
        self.RadarInit_MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: Conv 128x32x(W/8) -> Conv 256x16x(W/16)
        self.RadarInit_Conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.RadarInit_BatchNorm4 = nn.BatchNorm2d(256)
        self.RadarInit_MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Projection 1x1 convolution layer to align channels to the target latent dimension (D = 256)
        self.RadarInit_Projection = nn.Conv2d(
            in_channels=256,
            out_channels=RadarInit_LatentDim,
            kernel_size=1
        )

    # Forward pass mapping radar maps to features
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
