
# =============================================================================
# SECTION: Bounding Box IoU Calculation Utility
# =============================================================================
def compute_iou_metric(Metric_PredBox, Metric_TrueBox):
    # Metric_PredBox: numpy array of format [x_center, y_center, width, height]
    # Metric_TrueBox: numpy array of format [x_center, y_center, width, height]
    
    # Calculate top-left and bottom-right coordinates for predicted box
    Metric_Px1 = Metric_PredBox[0] - Metric_PredBox[2] / 2.0
    Metric_Py1 = Metric_PredBox[1] - Metric_PredBox[3] / 2.0
    Metric_Px2 = Metric_PredBox[0] + Metric_PredBox[2] / 2.0
    Metric_Py2 = Metric_PredBox[1] + Metric_PredBox[3] / 2.0
    
    # Calculate top-left and bottom-right coordinates for ground-truth box
    Metric_Tx1 = Metric_TrueBox[0] - Metric_TrueBox[2] / 2.0
    Metric_Ty1 = Metric_TrueBox[1] - Metric_TrueBox[3] / 2.0
    Metric_Tx2 = Metric_TrueBox[0] + Metric_TrueBox[2] / 2.0
    Metric_Ty2 = Metric_TrueBox[1] + Metric_TrueBox[3] / 2.0
    
    # Calculate intersection box bounds coordinates
    Metric_Ix1 = max(Metric_Px1, Metric_Tx1)
    Metric_Iy1 = max(Metric_Py1, Metric_Ty1)
    Metric_Ix2 = min(Metric_Px2, Metric_Tx2)
    Metric_Iy2 = min(Metric_Py2, Metric_Ty2)
    
    # Compute intersection area dimensions
    Metric_InterWidth = max(0.0, Metric_Ix2 - Metric_Ix1)
    Metric_InterHeight = max(0.0, Metric_Iy2 - Metric_Iy1)
    Metric_InterArea = Metric_InterWidth * Metric_InterHeight
    
    # Compute area of both predicted and ground-truth boxes
    Metric_PredArea = (Metric_Px2 - Metric_Px1) * (Metric_Py2 - Metric_Py1)
    Metric_TrueArea = (Metric_Tx2 - Metric_Tx1) * (Metric_Ty2 - Metric_Ty1)
    
    # Compute total union area with epsilon to prevent division by zero
    Metric_UnionArea = Metric_PredArea + Metric_TrueArea - Metric_InterArea + 1e-7
    
    # Calculate Intersection over Union
    Metric_IoU = Metric_InterArea / Metric_UnionArea
    
    # Return calculated float IoU score
    return float(Metric_IoU)
