# =============================================================================
# STUDENT-CODING-AGENT NOTICE:
# Style rules, including line-by-line comments and prefixed variable names,
# are applied internally in this detection head file.
# =============================================================================

# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import neural network modules from PyTorch
import torch.nn as nn

# =============================================================================
# SECTION: Vision & Fused Detection Head Module
# =============================================================================
class VisionDetectionHead(nn.Module):
    # Constructor mapping flattened spatial tokens to intermediate representations
    def __init__(self, HeadInit_NumTokens=49, HeadInit_LatentDim=256):
        # Invoke parent constructor
        super(VisionDetectionHead, self).__init__()
        
        # Shared MLP block mapping camera feature representation to target dimension (128)
        self.HeadInit_MLP = nn.Sequential(
            nn.Linear(HeadInit_NumTokens * HeadInit_LatentDim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Classification projection mapping features to 1 logit
        self.HeadInit_Classifier = nn.Linear(128, 1)
        
        # Regression projection mapping features to 4 normalized bounding box outputs
        self.HeadInit_Regressor = nn.Linear(128, 4)

    # Forward pass generating classifications and regressed coordinates
    def forward(self, HeadForward_FlattenedTokens):
        # Pass flattened tokens through shared MLP layers
        HeadForward_MLPFeatures = self.HeadInit_MLP(HeadForward_FlattenedTokens)
        # Generate target presence logit score
        HeadForward_ClassLogit = self.HeadInit_Classifier(HeadForward_MLPFeatures)
        # Generate bounding box location coordinates [x, y, w, h] normalized in [0, 1]
        HeadForward_BBoxPred = torch.sigmoid(self.HeadInit_Regressor(HeadForward_MLPFeatures))
        # Return predicted class logit score and regressed coordinate coordinates
        return HeadForward_ClassLogit, HeadForward_BBoxPred

# =============================================================================
# SECTION: Radar-Only Detection Head Module
# =============================================================================
class RadarDetectionHead(nn.Module):
    # Constructor mapping average pooled radar tokens to representations
    def __init__(self, HeadInit_LatentDim=256):
        # Invoke parent constructor
        super(RadarDetectionHead, self).__init__()
        
        # Shared MLP block mapping radar feature representations to target dimension (128)
        self.HeadInit_MLP = nn.Sequential(
            nn.Linear(HeadInit_LatentDim, 128),
            nn.ReLU()
        )
        
        # Classification projection mapping features to 1 logit
        self.HeadInit_Classifier = nn.Linear(128, 1)
        
        # Regression projection mapping features to 4 normalized bounding box outputs
        self.HeadInit_Regressor = nn.Linear(128, 4)

    # Forward pass generating predictions from pooled features
    def forward(self, HeadForward_PooledFeatures):
        # Pass features through shared MLP layers
        HeadForward_MLPFeatures = self.HeadInit_MLP(HeadForward_PooledFeatures)
        # Generate class logit prediction score
        HeadForward_ClassLogit = self.HeadInit_Classifier(HeadForward_MLPFeatures)
        # Generate bounding box location coordinates [x, y, w, h] normalized in [0, 1]
        HeadForward_BBoxPred = torch.sigmoid(self.HeadInit_Regressor(HeadForward_MLPFeatures))
        # Return predictions
        return HeadForward_ClassLogit, HeadForward_BBoxPred
