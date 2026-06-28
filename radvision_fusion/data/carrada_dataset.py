
# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import base Dataset class from PyTorch utilities
from torch.utils.data import Dataset
# Import numpy library for array allocations and mathematical simulation
import numpy as np
# Import random module to execute synchronized data augmentation decisions
import random

# =============================================================================
# SECTION: Custom Dataset representing CARRADA Simulation
# =============================================================================
class RadVisionDataset(Dataset):
    # Constructor method for dataset initialization
    def __init__(self, DatasetInit_NumSamples=100, DatasetInit_OcclusionProb=0.0):
        # Store total sample count
        self.DatasetInit_NumSamples = DatasetInit_NumSamples
        # Store probability of camera frame occlusion simulation
        self.DatasetInit_OcclusionProb = DatasetInit_OcclusionProb

    # Return total sample length of dataset
    def __len__(self):
        # Return saved number of samples
        return self.DatasetInit_NumSamples

    # Fetch a single multi-modal dictionary of synchronized tensors
    def __getitem__(self, DatasetGet_Index):
        # Set a deterministic numpy random seed based on sample index
        np.random.seed(DatasetGet_Index)
        
        # Simulate a 3-channel RGB Camera frame of size 3x224x224
        # We will generate a structured image: a dark background with a bright red circle object
        DatasetGet_CameraFrame = np.zeros((3, 224, 224), dtype=np.float32)
        # Bounding box center coordinates
        DatasetGet_CenterY = 112
        DatasetGet_CenterX = 112
        DatasetGet_Radius = 30
        
        # Draw red circle in the center of the simulated camera frame
        for Loop_Y in range(224):
            for Loop_X in range(224):
                if (Loop_Y - DatasetGet_CenterY)**2 + (Loop_X - DatasetGet_CenterX)**2 < DatasetGet_Radius**2:
                    # Set red channel value
                    DatasetGet_CameraFrame[0, Loop_Y, Loop_X] = 1.0
                    # Set green channel value
                    DatasetGet_CameraFrame[1, Loop_Y, Loop_X] = 0.1
                    # Set blue channel value
                    DatasetGet_CameraFrame[2, Loop_Y, Loop_X] = 0.1
        
        # Add tiny Gaussian noise to camera frame background
        DatasetGet_CameraFrame += np.random.normal(0, 0.05, DatasetGet_CameraFrame.shape).astype(np.float32)
        # Clip pixel intensities to [0, 1] bounds
        DatasetGet_CameraFrame = np.clip(DatasetGet_CameraFrame, 0.0, 1.0)
        
        # Check if the camera visual input should be occluded (fog/low-light simulation)
        if random.random() < self.DatasetInit_OcclusionProb:
            # Scale down and add high sensor noise to camera pixels to simulate fog occlusion
            DatasetGet_CameraFrame = DatasetGet_CameraFrame * 0.1 + np.random.normal(0, 0.2, DatasetGet_CameraFrame.shape).astype(np.float32)
            # Clip pixel intensities to [0, 1] bounds
            DatasetGet_CameraFrame = np.clip(DatasetGet_CameraFrame, 0.0, 1.0)
            
        # Simulate a 1-channel Range-Doppler (RD) radar heatmap of size 1x256x64
        # Target signature placed at center bin [128, 32]
        DatasetGet_RadarRD = np.random.normal(0, 0.1, (1, 256, 64)).astype(np.float32)
        DatasetGet_RadarRD[0, 128, 32] = 2.5
        
        # Simulate a 1-channel Range-Angle (RA) radar heatmap of size 1x256x256
        # Target signature placed at center bin [128, 128]
        DatasetGet_RadarRA = np.random.normal(0, 0.1, (1, 256, 256)).astype(np.float32)
        DatasetGet_RadarRA[0, 128, 128] = 2.5
        
        # Ground truth bounding box coords: [x_center, y_center, width, height] (normalized)
        DatasetGet_BBox = np.array([0.5, 0.5, 0.27, 0.27], dtype=np.float32)
        
        # Ground truth classification label (1.0 = target object exists)
        DatasetGet_ClassLabel = np.array([1.0], dtype=np.float32)
        
        # Apply synchronized data augmentations: random horizontal flip
        if random.random() > 0.5:
            # Flip camera frame along horizontal width dimension
            DatasetGet_CameraFrame = np.flip(DatasetGet_CameraFrame, axis=2).copy()
            # Flip Range-Doppler speed channels horizontally
            DatasetGet_RadarRD = np.flip(DatasetGet_RadarRD, axis=2).copy()
            # Flip Range-Angle angular channels horizontally
            DatasetGet_RadarRA = np.flip(DatasetGet_RadarRA, axis=2).copy()
            # Invert the normalized x center coordinate of our box target
            DatasetGet_BBox[0] = 1.0 - DatasetGet_BBox[0]

        # Convert camera frame to PyTorch float tensor
        DatasetGet_CameraTensor = torch.tensor(DatasetGet_CameraFrame)
        # Convert Range-Doppler map to PyTorch float tensor
        DatasetGet_RadarRDTensor = torch.tensor(DatasetGet_RadarRD)
        # Convert Range-Angle map to PyTorch float tensor
        DatasetGet_RadarRATensor = torch.tensor(DatasetGet_RadarRA)
        # Convert bounding box array to PyTorch float tensor
        DatasetGet_BBoxTensor = torch.tensor(DatasetGet_BBox)
        # Convert class label array to PyTorch float tensor
        DatasetGet_ClassTensor = torch.tensor(DatasetGet_ClassLabel)
        
        # Return structured sample dictionary
        return {
            "camera": DatasetGet_CameraTensor,
            "radar_rd": DatasetGet_RadarRDTensor,
            "radar_ra": DatasetGet_RadarRATensor,
            "bbox": DatasetGet_BBoxTensor,
            "label": DatasetGet_ClassTensor
        }
