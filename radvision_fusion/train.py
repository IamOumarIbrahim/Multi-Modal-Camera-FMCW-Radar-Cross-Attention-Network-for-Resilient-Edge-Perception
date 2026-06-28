# =============================================================================
# STUDENT-CODING-AGENT NOTICE:
# Style rules, including line-by-line comments and prefixed variable names,
# are applied internally in this training executable script file.
# =============================================================================

# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import PyTorch core deep learning library
import torch
# Import DataLoader class from PyTorch utilities
from torch.utils.data import DataLoader
# Import configurations from config file
from radvision_fusion.config.config import CONFIG_LATENT_DIM, CONFIG_BATCH_SIZE, CONFIG_EPOCHS
# Import simulated CARRADA dataset class
from radvision_fusion.data.carrada_dataset import RadVisionDataset
# Import unified fusion network model
from radvision_fusion.models.radvision_model import RadVisionFusionModel
# Import binary focal loss and complete IoU loss functions
from radvision_fusion.utils.losses import compute_focal_loss, compute_ciou_loss

# =============================================================================
# SECTION: Model Training Function
# =============================================================================
def train_model(Train_ModelInstance, Train_DataLoaderInstance, Train_Epochs=5):
    # Check if GPU acceleration is available and map device accordingly
    Train_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to selected device
    Train_ModelInstance.to(Train_Device)
    # Put model in active training state
    Train_ModelInstance.train()
    
    # Instantiate standard Adam optimizer for model parameters
    Train_Optimizer = torch.optim.Adam(Train_ModelInstance.parameters(), lr=0.001)
    # Initialize PyTorch GradScaler for float16 mixed precision training
    Train_Scaler = torch.cuda.amp.GradScaler(enabled=(Train_Device.type == "cuda"))

    # Loop through training epochs
    for Loop_EpochIndex in range(Train_Epochs):
        # Initialize running loss accumulator for debugging
        Train_RunningLoss = 0.0
        
        # Process batches from DataLoader
        for Loop_BatchIndex, Train_BatchData in enumerate(Train_DataLoaderInstance):
            # Extract inputs and transfer to device
            Train_Camera = Train_BatchData["camera"].to(Train_Device)
            Train_RadarRD = Train_BatchData["radar_rd"].to(Train_Device)
            Train_RadarRA = Train_BatchData["radar_ra"].to(Train_Device)
            # Extract labels and transfer to device
            Train_BBoxTrue = Train_BatchData["bbox"].to(Train_Device)
            Train_LabelTrue = Train_BatchData["label"].to(Train_Device)

            # Reset optimizer gradients
            Train_Optimizer.zero_grad()

            # Execute forward pass within autocast scope to utilize float16 acceleration
            with torch.cuda.amp.autocast(enabled=(Train_Device.type == "cuda")):
                # Generate predictions
                Train_ClassLogit, Train_BBoxPred, _ = Train_ModelInstance(
                    Train_Camera, Train_RadarRD, Train_RadarRA
                )
                
                # Compute Focal Loss for class confidence
                Train_Focal = compute_focal_loss(Train_ClassLogit, Train_LabelTrue)
                # Compute CIoU Loss for bounding box regressor
                Train_CIoU = compute_ciou_loss(Train_BBoxPred, Train_BBoxTrue)
                # Total loss with equal weighting coefficients alpha = 1.0, beta = 1.0
                Train_TotalLoss = Train_Focal + Train_CIoU

            # Run backward pass with scaler scaling to prevent underflow
            Train_Scaler.scale(Train_TotalLoss).backward()
            # Update parameters via scaler step
            Train_Scaler.step(Train_Optimizer)
            # Update scale factor for next iteration
            Train_Scaler.update()

            # Accumulate running loss value
            Train_RunningLoss += Train_TotalLoss.item()

        # Print training progress metrics per epoch
        print(f"Epoch {Loop_EpochIndex + 1}/{Train_Epochs} - Mean Loss: {Train_RunningLoss / len(Train_DataLoaderInstance):.4f}")

# =============================================================================
# SECTION: Main Script Entry Point
# =============================================================================
if __name__ == "__main__":
    # Print main task header banner
    print("#############################################")
    print("# EXECUTING: RadVision Model Training")
    print("#############################################\n")

    # Create dataset instance with simulated occlusion probability
    Main_TrainDataset = RadVisionDataset(DatasetInit_NumSamples=32, DatasetInit_OcclusionProb=0.2)
    # Create dataloader utilizing configurations
    Main_TrainLoader = DataLoader(Main_TrainDataset, batch_size=CONFIG_BATCH_SIZE, shuffle=True)

    # Instantiate the unified multi-modal fusion model
    Main_Model = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="fused")
    
    # Train the unified network model
    train_model(Main_Model, Main_TrainLoader, Train_Epochs=CONFIG_EPOCHS)
    
    # Save the trained model weights locally
    torch.save(Main_Model.state_dict(), "radvision_model.pth")
    
    # Print execution exit confirmation banner
    print("\n=============================================")
    print("[SUCCESS] Model Saved to radvision_model.pth")
    print("=============================================")
