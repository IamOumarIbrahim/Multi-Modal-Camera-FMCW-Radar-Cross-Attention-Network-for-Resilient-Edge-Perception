
# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import base neural network module class from PyTorch
import torch.nn as nn
# Import torchvision models to extract standard backbones
import torchvision

# =============================================================================
# SECTION: Vision Encoder Branch (Ec)
# =============================================================================
class VisionEncoder(nn.Module):
    # Constructor for initializing the vision encoder parameters
    def __init__(self, VisionInit_LatentDim=256):
        # Invoke parent constructor
        super(VisionEncoder, self).__init__()
        
        # Instantiate a standard ResNet18 model without pretrained weights to allow offline execution
        VisionInit_ResNet = torchvision.models.resnet18(weights=None)
        
        # Extract convolutional backbone layers (up to Layer 4)
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
        
        # 1x1 projection convolution to map ResNet-18 channels (512) to latent dimension (D = 256)
        self.VisionInit_Projection = nn.Conv2d(
            in_channels=512,
            out_channels=VisionInit_LatentDim,
            kernel_size=1
        )

    # Forward pass mapping camera frame tensor to latent space
    def forward(self, VisionForward_InputTensor):
        # Extract features from backbone layers: [Batch, 3, 224, 224] -> [Batch, 512, 7, 7]
        VisionForward_Features = self.VisionInit_Backbone(VisionForward_InputTensor)
        # Apply 1x1 projection convolution: [Batch, 512, 7, 7] -> [Batch, 256, 7, 7]
        VisionForward_Projected = self.VisionInit_Projection(VisionForward_Features)
        # Return projected features
        return VisionForward_Projected
