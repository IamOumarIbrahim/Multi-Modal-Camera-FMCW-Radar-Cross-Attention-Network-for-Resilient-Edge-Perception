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
# Import os module for file path validations
import os
# Import json module to read dataset annotation configurations
import json
# Import PIL Image for camera frame loading and scaling
from PIL import Image

# =============================================================================
# SECTION: Custom Dataset representing CARRADA Real & Simulation Fallback
# =============================================================================
class RadVisionDataset(Dataset):
    # Constructor method for dataset initialization
    def __init__(self, DatasetInit_NumSamples=100, DatasetInit_OcclusionProb=0.0, DatasetInit_CarradaPath="D:/cold storage/CARRADA"):
        # Store total sample count
        self.DatasetInit_NumSamples = DatasetInit_NumSamples
        # Store probability of camera frame occlusion simulation
        self.DatasetInit_OcclusionProb = DatasetInit_OcclusionProb
        # Store root data path to the real CARRADA dataset
        self.DatasetInit_CarradaPath = DatasetInit_CarradaPath
        # Flag indicating if dataset is operating on real files
        self.DatasetInit_UseReal = False
        
        # Define absolute path to target annotations metadata file
        self.DatasetInit_AnnFile = os.path.join(self.DatasetInit_CarradaPath, "annotations.json")
        
        # Check if the CARRADA directory and annotations.json exist to enable real mode
        if os.path.exists(self.DatasetInit_CarradaPath) and os.path.exists(self.DatasetInit_AnnFile):
            try:
                # Open and parse metadata annotations file
                with open(self.DatasetInit_AnnFile, "r") as JSONFile:
                    self.DatasetInit_Annotations = json.load(JSONFile)
                
                # Build structural list of valid synchronized samples
                self.DatasetInit_Samples = []
                for SeqName, Frames in self.DatasetInit_Annotations.items():
                    for FrameId, Ann in Frames.items():
                        # Resolve absolute paths for image and radar heatmap files
                        ImgPath = os.path.join(self.DatasetInit_CarradaPath, SeqName, "camera_images", f"{FrameId}.jpg")
                        RdPath = os.path.join(self.DatasetInit_CarradaPath, SeqName, "range_doppler", f"{FrameId}.npy")
                        RaPath = os.path.join(self.DatasetInit_CarradaPath, SeqName, "range_angle", f"{FrameId}.npy")
                        
                        # Verify that all corresponding sensor files exist
                        if os.path.exists(ImgPath) and os.path.exists(RdPath) and os.path.exists(RaPath):
                            self.DatasetInit_Samples.append({
                                "seq_name": SeqName,
                                "frame_id": FrameId,
                                "img_path": ImgPath,
                                "rd_path": RdPath,
                                "ra_path": RaPath,
                                "bbox": Ann.get("box", [0.5, 0.5, 0.27, 0.27]),
                                "label": Ann.get("label", 1.0)
                            })
                
                # Enable real mode if at least one complete sample is validated
                if len(self.DatasetInit_Samples) > 0:
                    self.DatasetInit_UseReal = True
                    # Limit sample size to requested dataset bounds
                    if self.DatasetInit_NumSamples < len(self.DatasetInit_Samples):
                        self.DatasetInit_Samples = self.DatasetInit_Samples[:self.DatasetInit_NumSamples]
                    else:
                        self.DatasetInit_NumSamples = len(self.DatasetInit_Samples)
                    print(f"RadVisionDataset: Successfully loaded {self.DatasetInit_NumSamples} real CARRADA samples.")
            except Exception as e:
                print(f"RadVisionDataset Warning: Error parsing real dataset ({e}). Falling back to simulation.")
        
        # Print fallback notice if simulated mode is active
        if not self.DatasetInit_UseReal:
            print("RadVisionDataset: Real CARRADA path not found. Operating in SIMULATED fallback mode.")

    # Return total sample length of dataset
    def __len__(self):
        # Return saved number of samples
        return self.DatasetInit_NumSamples

    # Fetch a single multi-modal dictionary of synchronized tensors
    def __getitem__(self, DatasetGet_Index):
        if self.DatasetInit_UseReal:
            # -----------------------------------------------------------------
            # MODE A: Load Real CARRADA Dataset Tensors
            # -----------------------------------------------------------------
            SampleInfo = self.DatasetInit_Samples[DatasetGet_Index]
            
            # Load raw RGB camera image and scale to target size [224, 224]
            Img = Image.open(SampleInfo["img_path"]).convert("RGB").resize((224, 224))
            # Format image to PyTorch shape channels-first [3, 224, 224] and normalize to [0, 1]
            DatasetGet_CameraFrame = np.array(Img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            
            # Load range-doppler and range-angle radar arrays
            RadarRD = np.load(SampleInfo["rd_path"]).astype(np.float32)
            RadarRA = np.load(SampleInfo["ra_path"]).astype(np.float32)
            
            # Expand dimensions to append single-channel representation if needed
            if len(RadarRD.shape) == 2:
                RadarRD = np.expand_dims(RadarRD, axis=0)
            if len(RadarRA.shape) == 2:
                RadarRA = np.expand_dims(RadarRA, axis=0)
                
            # Extract annotation targets
            DatasetGet_BBox = np.array(SampleInfo["bbox"], dtype=np.float32)
            DatasetGet_ClassLabel = np.array([float(SampleInfo["label"])], dtype=np.float32)
        else:
            # -----------------------------------------------------------------
            # MODE B: Simulated Fallback Mode
            # -----------------------------------------------------------------
            np.random.seed(DatasetGet_Index)
            DatasetGet_CameraFrame = np.zeros((3, 224, 224), dtype=np.float32)
            DatasetGet_CenterY = 112
            DatasetGet_CenterX = 112
            DatasetGet_Radius = 30
            for Loop_Y in range(224):
                for Loop_X in range(224):
                    if (Loop_Y - DatasetGet_CenterY)**2 + (Loop_X - DatasetGet_CenterX)**2 < DatasetGet_Radius**2:
                        DatasetGet_CameraFrame[0, Loop_Y, Loop_X] = 1.0
                        DatasetGet_CameraFrame[1, Loop_Y, Loop_X] = 0.1
                        DatasetGet_CameraFrame[2, Loop_Y, Loop_X] = 0.1
            DatasetGet_CameraFrame += np.random.normal(0, 0.05, DatasetGet_CameraFrame.shape).astype(np.float32)
            DatasetGet_CameraFrame = np.clip(DatasetGet_CameraFrame, 0.0, 1.0)
            
            RadarRD = np.random.normal(0, 0.1, (1, 256, 64)).astype(np.float32)
            RadarRD[0, 128, 32] = 2.5
            RadarRA = np.random.normal(0, 0.1, (1, 256, 256)).astype(np.float32)
            RadarRA[0, 128, 128] = 2.5
            DatasetGet_BBox = np.array([0.5, 0.5, 0.27, 0.27], dtype=np.float32)
            DatasetGet_ClassLabel = np.array([1.0], dtype=np.float32)

        # Apply occlusion to camera frame if requested
        if random.random() < self.DatasetInit_OcclusionProb:
            DatasetGet_CameraFrame = DatasetGet_CameraFrame * 0.1 + np.random.normal(0, 0.2, DatasetGet_CameraFrame.shape).astype(np.float32)
            DatasetGet_CameraFrame = np.clip(DatasetGet_CameraFrame, 0.0, 1.0)

        # Apply horizontal flip augmentation
        if random.random() > 0.5:
            DatasetGet_CameraFrame = np.flip(DatasetGet_CameraFrame, axis=2).copy()
            RadarRD = np.flip(RadarRD, axis=2).copy()
            RadarRA = np.flip(RadarRA, axis=2).copy()
            DatasetGet_BBox[0] = 1.0 - DatasetGet_BBox[0]

        return {
            "camera": torch.tensor(DatasetGet_CameraFrame),
            "radar_rd": torch.tensor(RadarRD),
            "radar_ra": torch.tensor(RadarRA),
            "bbox": torch.tensor(DatasetGet_BBox),
            "label": torch.tensor(DatasetGet_ClassLabel)
        }
