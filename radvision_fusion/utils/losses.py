
# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import neural network functional operations from PyTorch
import torch.nn.functional as F
# Import numpy library for math constants (like Pi)
import numpy as np

# =============================================================================
# SECTION: Binary Focal Loss Implementation
# =============================================================================
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

# =============================================================================
# SECTION: Complete Intersection over Union (CIoU) Loss Implementation
# =============================================================================
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
