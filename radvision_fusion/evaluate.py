
# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import DataLoader utility from PyTorch
from torch.utils.data import DataLoader
# Import numpy for array manipulation and mean calculations
import numpy as np
# Import configuration constants from local config file
from radvision_fusion.config.config import CONFIG_LATENT_DIM, CONFIG_BATCH_SIZE, CONFIG_EPOCHS
# Import simulated CARRADA dataset
from radvision_fusion.data.carrada_dataset import RadVisionDataset
# Import unified fusion network model
from radvision_fusion.models.radvision_model import RadVisionFusionModel
# Import training helper function from train script
from radvision_fusion.train import train_model

# =============================================================================
# SECTION: Model Evaluation Function (Ablation testing)
# =============================================================================
def evaluate_model(Eval_ModelInstance, Eval_DataLoaderInstance):
    # Check if GPU acceleration is available
    Eval_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Put model in evaluation state
    Eval_ModelInstance.eval()
    # Move model to device
    Eval_ModelInstance.to(Eval_Device)
    
    # Initialize lists to store computed IoU values
    Eval_IoUList = []

    # Disable gradient tracking for speed and memory efficiency
    with torch.no_grad():
        # Process batches from DataLoader
        for Eval_BatchData in Eval_DataLoaderInstance:
            # Transfer input variables to device
            Eval_Camera = Eval_BatchData["camera"].to(Eval_Device)
            Eval_RadarRD = Eval_BatchData["radar_rd"].to(Eval_Device)
            Eval_RadarRA = Eval_BatchData["radar_ra"].to(Eval_Device)
            Eval_BBoxTrue = Eval_BatchData["bbox"].to(Eval_Device)

            # Generate predictions
            _, Eval_BBoxPred, _ = Eval_ModelInstance(
                Eval_Camera, Eval_RadarRD, Eval_RadarRA
            )

            # Calculate IoU for each predicted box in the batch
            Eval_PredX1 = Eval_BBoxPred[:, 0] - Eval_BBoxPred[:, 2] / 2.0
            Eval_PredY1 = Eval_BBoxPred[:, 1] - Eval_BBoxPred[:, 3] / 2.0
            Eval_PredX2 = Eval_BBoxPred[:, 0] + Eval_BBoxPred[:, 2] / 2.0
            Eval_PredY2 = Eval_BBoxPred[:, 1] + Eval_BBoxPred[:, 3] / 2.0

            Eval_TrueX1 = Eval_BBoxTrue[:, 0] - Eval_BBoxTrue[:, 2] / 2.0
            Eval_TrueY1 = Eval_BBoxTrue[:, 1] - Eval_BBoxTrue[:, 3] / 2.0
            Eval_TrueX2 = Eval_BBoxTrue[:, 0] + Eval_BBoxTrue[:, 2] / 2.0
            Eval_TrueY2 = Eval_BBoxTrue[:, 1] + Eval_BBoxTrue[:, 3] / 2.0

            Eval_PredArea = (Eval_PredX2 - Eval_PredX1).clamp(min=0) * (Eval_PredY2 - Eval_PredY1).clamp(min=0)
            Eval_TrueArea = (Eval_TrueX2 - Eval_TrueX1).clamp(min=0) * (Eval_TrueY2 - Eval_TrueY1).clamp(min=0)

            Eval_InterX1 = torch.max(Eval_PredX1, Eval_TrueX1)
            Eval_InterY1 = torch.max(Eval_PredY1, Eval_TrueY1)
            Eval_InterX2 = torch.min(Eval_PredX2, Eval_TrueX2)
            Eval_InterY2 = torch.min(Eval_PredY2, Eval_TrueY2)

            Eval_InterArea = (Eval_InterX2 - Eval_InterX1).clamp(min=0) * (Eval_InterY2 - Eval_InterY1).clamp(min=0)
            Eval_UnionArea = Eval_PredArea + Eval_TrueArea - Eval_InterArea + 1e-7
            Eval_IoU = Eval_InterArea / Eval_UnionArea
            
            # Extend lists with computed batch IoU values
            Eval_IoUList.extend(Eval_IoU.cpu().numpy().tolist())

    # Return average mean IoU over whole validation set
    return np.mean(Eval_IoUList)

# =============================================================================
# SECTION: Main Script Entry Point (Ablation Studies compilation)
# =============================================================================
if __name__ == "__main__":
    print("#############################################")
    print("# EXECUTING: Ablation Benchmarking Suite")
    print("#############################################\n")

    # Create datasets
    print("Creating Simulated Datasets...")
    Main_TrainDataset = RadVisionDataset(DatasetInit_NumSamples=32, DatasetInit_OcclusionProb=0.2)
    Main_TestDatasetClean = RadVisionDataset(DatasetInit_NumSamples=16, DatasetInit_OcclusionProb=0.0)
    Main_TestDatasetOccluded = RadVisionDataset(DatasetInit_NumSamples=16, DatasetInit_OcclusionProb=1.0)

    # Create data loaders
    Main_TrainLoader = DataLoader(Main_TrainDataset, batch_size=CONFIG_BATCH_SIZE, shuffle=True)
    Main_TestLoaderClean = DataLoader(Main_TestDatasetClean, batch_size=CONFIG_BATCH_SIZE, shuffle=False)
    Main_TestLoaderOccluded = DataLoader(Main_TestDatasetOccluded, batch_size=CONFIG_BATCH_SIZE, shuffle=False)

    # 1. Train and Evaluate Vision-Only Branch
    print("\n---------------------------------------------")
    print("Training Vision-Only Model...")
    print("---------------------------------------------")
    Main_VisionOnlyModel = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="vision_only")
    train_model(Main_VisionOnlyModel, Main_TrainLoader, Train_Epochs=CONFIG_EPOCHS)
    Main_VisionCleanIoU = evaluate_model(Main_VisionOnlyModel, Main_TestLoaderClean)
    Main_VisionOccludedIoU = evaluate_model(Main_VisionOnlyModel, Main_TestLoaderOccluded)

    # 2. Train and Evaluate Radar-Only Branch
    print("\n---------------------------------------------")
    print("Training Radar-Only Model...")
    print("---------------------------------------------")
    Main_RadarOnlyModel = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="radar_only")
    train_model(Main_RadarOnlyModel, Main_TrainLoader, Train_Epochs=CONFIG_EPOCHS)
    Main_RadarCleanIoU = evaluate_model(Main_RadarOnlyModel, Main_TestLoaderClean)
    Main_RadarOccludedIoU = evaluate_model(Main_RadarOnlyModel, Main_TestLoaderOccluded)

    # 3. Train and Evaluate Full RadVision Fused Model
    print("\n---------------------------------------------")
    print("Training RadVision (Fused) Model...")
    print("---------------------------------------------")
    Main_FusedModel = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="fused")
    train_model(Main_FusedModel, Main_TrainLoader, Train_Epochs=CONFIG_EPOCHS)
    Main_FusedCleanIoU = evaluate_model(Main_FusedModel, Main_TestLoaderClean)
    Main_FusedOccludedIoU = evaluate_model(Main_FusedModel, Main_TestLoaderOccluded)

    # Print Ablation Table Results
    print("\n=============================================")
    print("[ABLATION] Modality Comparison Table")
    print("=============================================")
    print(f"{'Model':<18} | {'Camera':<8} | {'Radar':<8} | {'Test Clean IoU':<14} | {'Test Occluded IoU':<17}")
    print("-" * 75)
    print(f"{'Vision-Only':<18} | {'Active':<8} | {'Ignored':<8} | {Main_VisionCleanIoU:<14.4f} | {Main_VisionOccludedIoU:<17.4f}")
    print(f"{'Radar-Only':<18} | {'Ignored':<8} | {'Active':<8} | {Main_RadarCleanIoU:<14.4f} | {Main_RadarOccludedIoU:<17.4f}")
    print(f"{'RadVision (Fused)':<18} | {'Active':<8} | {'Active':<8} | {Main_FusedCleanIoU:<14.4f} | {Main_FusedOccludedIoU:<17.4f}")
    print("=============================================\n")
