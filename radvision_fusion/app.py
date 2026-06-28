# =============================================================================
# STUDENT-CODING-AGENT NOTICE:
# Style rules, including line-by-line comments and prefixed variable names,
# are applied internally in this interactive Streamlit application file.
# =============================================================================

# =============================================================================
# SECTION: Import Libraries
# =============================================================================
# Import Streamlit library for interactive web dashboard UI
import streamlit as st
# Import PyTorch core deep learning library
import torch
# Import DataLoader class from PyTorch utilities
from torch.utils.data import DataLoader
# Import numpy library for array processing
import numpy as np
# Import random module for random augmentation decisions
import random
# Import matplotlib for rendering images and figures
import matplotlib.pyplot as plt
# Import patches from matplotlib to draw bounding boxes
import matplotlib.patches as patches

# Import configuration constants from local config file
from radvision_fusion.config.config import CONFIG_LATENT_DIM
# Import simulated CARRADA dataset from data folder
from radvision_fusion.data.carrada_dataset import RadVisionDataset
# Import unified cross-attention network model
from radvision_fusion.models.radvision_model import RadVisionFusionModel
# Import custom focal and CIoU losses
from radvision_fusion.utils.losses import compute_focal_loss, compute_ciou_loss
# Import IoU metrics calculator
from radvision_fusion.utils.metrics import compute_iou_metric

# =============================================================================
# SECTION: Model Training Function (Caching resource to avoid repeat training)
# =============================================================================
@st.cache_resource
def train_and_cache_models():
    # Instantiate dataset and dataloader
    DatasetInstance = RadVisionDataset(DatasetInit_NumSamples=32)
    DataLoaderInstance = DataLoader(DatasetInstance, batch_size=4, shuffle=True)
    
    # Define models
    VisionModel = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="vision_only")
    RadarModel = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="radar_only")
    FusedModel = RadVisionFusionModel(ModelInit_LatentDim=CONFIG_LATENT_DIM, ModelInit_Mode="fused")
    
    # Train each model briefly for 5 epochs
    for ModelItem, ModelName in [(VisionModel, "Vision-Only"), (RadarModel, "Radar-Only"), (FusedModel, "RadVision (Fused)")]:
        ModelItem.train()
        Optimizer = torch.optim.Adam(ModelItem.parameters(), lr=0.001)
        for Epoch in range(5):
            for Batch in DataLoaderInstance:
                Optimizer.zero_grad()
                ClassLogit, BBoxPred, _ = ModelItem(
                    Batch["camera"], Batch["radar_rd"], Batch["radar_ra"]
                )
                Focal = compute_focal_loss(ClassLogit, Batch["label"])
                CIoU = compute_ciou_loss(BBoxPred, Batch["bbox"])
                TotalLoss = Focal + CIoU
                TotalLoss.backward()
                Optimizer.step()
        ModelItem.eval()
        
    return VisionModel, RadarModel, FusedModel

# =============================================================================
# SECTION: Streamlit UI Execution Layout
# =============================================================================
# Configure general Streamlit page properties
st.set_page_config(
    page_title="RadVision-Fusion: Resilient Multi-Modal Perception",
    page_icon="📡",
    layout="wide"
)

# Inject custom CSS for premium styling cards
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle Banner
st.title("📡 RadVision-Fusion")
st.subheader("Multi-Modal Camera–FMCW Radar Cross-Attention Network for Resilient Edge Perception")
st.markdown("---")

# Load trained models from cache
VisionModel, RadarModel, FusedModel = train_and_cache_models()

# Initialize Dataset Instance
DatasetInstance = RadVisionDataset(DatasetInit_NumSamples=20)

# Sidebar layout for user control inputs
st.sidebar.header("🎛️ Simulation Controls")
SampleIndex = st.sidebar.slider("Select Sample Index", 0, 19, 0)
OcclusionLevel = st.sidebar.slider("Camera Visual Occlusion Level", 0.0, 1.0, 0.5, step=0.1)

# Fetch selected sample from dataset
SampleData = DatasetInstance[SampleIndex]
CameraFrame = SampleData["camera"].clone()
RadarRD = SampleData["radar_rd"].clone()
RadarRA = SampleData["radar_ra"].clone()
BBoxTrue = SampleData["bbox"].numpy()

# Apply user occlusion to the camera frame (scale RGB channels down and add noise)
if OcclusionLevel > 0.0:
    # Scale down camera pixel intensity values to simulate low-light / fog blocking
    CameraFrame = CameraFrame * (1.0 - OcclusionLevel)
    # Add random Gaussian noise representing smoke / camera sensor corruption
    CameraFrame += torch.randn(CameraFrame.shape) * OcclusionLevel * 0.2
    # Clip pixels to keep them in [0, 1] range
    CameraFrame = torch.clamp(CameraFrame, 0.0, 1.0)

# Execute inference using all three models
with torch.no_grad():
    # Model 1: Vision-Only
    _, BBoxVisionPred, _ = VisionModel(
        CameraFrame.unsqueeze(0), RadarRD.unsqueeze(0), RadarRA.unsqueeze(0)
    )
    # Model 2: Radar-Only
    _, BBoxRadarPred, _ = RadarModel(
        CameraFrame.unsqueeze(0), RadarRD.unsqueeze(0), RadarRA.unsqueeze(0)
    )
    # Model 3: RadVision (Fused) with attention matrix
    _, BBoxFusedPred, AttnWeights = FusedModel(
        CameraFrame.unsqueeze(0), RadarRD.unsqueeze(0), RadarRA.unsqueeze(0)
    )

# Calculate metrics for the UI cards
IoUVision = compute_iou_metric(BBoxVisionPred[0].numpy(), BBoxTrue)
IoURadar = compute_iou_metric(BBoxRadarPred[0].numpy(), BBoxTrue)
IoUFused = compute_iou_metric(BBoxFusedPred[0].numpy(), BBoxTrue)

# Main Dashboard Columns: Raw Feeds
st.header("🖼️ Raw Sensor Feeds & Modality Visualizations")
ColFeed1, ColFeed2, ColFeed3 = st.columns(3)

with ColFeed1:
    st.subheader("📸 Camera Feed (RGB)")
    # Render camera image using matplotlib with bounding box overlays
    FigCam, AxCam = plt.subplots(figsize=(4, 4))
    # Permute shape to standard matplotlib HWC format
    CamImageNP = CameraFrame.permute(1, 2, 0).numpy()
    AxCam.imshow(CamImageNP)
    AxCam.set_title("Synchronized Video Frame")
    AxCam.axis("off")
    
    # Add Ground Truth Bounding Box in Green
    GTBoxPatch = patches.Rectangle(
        ((BBoxTrue[0] - BBoxTrue[2]/2)*224, (BBoxTrue[1] - BBoxTrue[3]/2)*224),
        BBoxTrue[2]*224, BBoxTrue[3]*224,
        linewidth=2, edgecolor="g", facecolor="none", label="Ground Truth"
    )
    AxCam.add_patch(GTBoxPatch)
    
    # Add Predicted Bounding Box in Yellow (from Fused Model)
    PredBoxNP = BBoxFusedPred[0].numpy()
    FusedBoxPatch = patches.Rectangle(
        ((PredBoxNP[0] - PredBoxNP[2]/2)*224, (PredBoxNP[1] - PredBoxNP[3]/2)*224),
        PredBoxNP[2]*224, PredBoxNP[3]*224,
        linewidth=2, edgecolor="yellow", facecolor="none", label="Fused Prediction"
    )
    AxCam.add_patch(FusedBoxPatch)
    AxCam.legend(loc="upper right")
    
    st.pyplot(FigCam)

with ColFeed2:
    st.subheader("📈 Range-Doppler Heatmap")
    # Render RD Map using matplotlib viridis colormap
    FigRD, AxRD = plt.subplots(figsize=(4, 4))
    RDNP = RadarRD[0].numpy()
    # Crop around the central signal area for visual aesthetic clarity
    AxRD.imshow(RDNP[96:160, 16:48], cmap="viridis", aspect="auto")
    AxRD.set_title("Range-Doppler Spectrum")
    AxRD.set_xlabel("Doppler Speed Bin")
    AxRD.set_ylabel("Range Bin")
    st.pyplot(FigRD)

with ColFeed3:
    st.subheader("📐 Range-Angle Heatmap")
    # Render RA Map using matplotlib magma colormap
    FigRA, AxRA = plt.subplots(figsize=(4, 4))
    RANP = RadarRA[0].numpy()
    # Crop around the central signal area for visual aesthetic clarity
    AxRA.imshow(RANP[96:160, 96:160], cmap="magma", aspect="auto")
    AxRA.set_title("Range-Angle Cartesian Projection")
    AxRA.set_xlabel("Angular Bin")
    AxRA.set_ylabel("Range Bin")
    st.pyplot(FigRA)

st.markdown("---")

# Metrics Summary Cards
st.header("📊 Real-Time Bounding Box IoU Metrics")
ColM1, ColM2, ColM3 = st.columns(3)

with ColM1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vision-Only Model</div>
        <div class="metric-value">{IoUVision:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with ColM2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Radar-Only Model</div>
        <div class="metric-value">{IoURadar:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with ColM3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #4ade80;">{IoUFused:.4f}</div>
        <div class="metric-label">RadVision (Fused Model)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Cross-Attention Weights Visualization
st.header("🧠 Cross-Attention Matrix Heatmap")
st.markdown("Visualizes attention routing from Camera Queries ($Q$, 49 spatial patches) to Radar Keys/Values ($K/V$, 320 patches).")

if AttnWeights is not None:
    # Extract weights tensor: [Batch, 49, 320] -> [49, 320]
    AttnMatrix = AttnWeights[0].numpy()
    
    FigAttn, AxAttn = plt.subplots(figsize=(10, 4))
    ImAttn = AxAttn.imshow(AttnMatrix, cmap="inferno", aspect="auto")
    AxAttn.set_title("Cross-Attention Alignment Weights (Camera Query vs Combined Radar Key)")
    AxAttn.set_xlabel("Combined Radar Tokens (0-63: RD, 64-319: RA)")
    AxAttn.set_ylabel("Camera Token Index (7x7 Spatial Grid)")
    FigAttn.colorbar(ImAttn, ax=AxAttn)
    st.pyplot(FigAttn)
else:
    st.info("Cross-Attention weights are only available in Fused mode.")
