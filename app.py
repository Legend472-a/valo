import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --------------------- Config ---------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# Force CPU for Streamlit Cloud to prevent memory/CUDA errors
device = torch.device("cpu")
LABEL_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------------------- Model Architecture ---------------------
class TransferLearningResNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Use weights=None (replaces deprecated pretrained=False)
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# --------------------- Resource Loading ---------------------
@st.cache_resource
def load_model():
    model_path = "resnet_model.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file `{model_path}` not found in repository!")
        st.stop()
    
    try:
        # 1. Initialize model architecture
        model = TransferLearningResNet(num_classes=4)
        
        # 2. Load weights strictly to CPU
        # weights_only=True is a security best practice for newer Torch versions
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 3. Load state and set to evaluation mode
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Initialize the model
model = load_model()

# --------------------- Image Processing ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------------------- Main UI ---------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("---")

# Layout: Sidebar for instructions, Main for app
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload a brain MRI image (axial view preferred).")
    st.write("2. The model will analyze the image and predict the tumor type.")
    st.info("Note: This is an AI demo and not a clinical diagnosis tool.")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose MRI Image (JPG / PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded MRI Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Analysis")
        with st.spinner("Analyzing MRI data..."):
            # Prepare tensor
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                confidence, pred_idx = torch.max(probs, 0)
            
            predicted_class = LABEL_NAMES[pred_idx.item()]
            confidence_pct = confidence.item() * 100

            # Visual Result
            if predicted_class == "notumor":
                st.success(f"**RESULT: NO TUMOR DETECTED** ({confidence_pct:.1f}%)")
            else:
                st.error(f"**RESULT: {predicted_class.upper()} TUMOR DETECTED**")
                st.metric("Confidence Score", f"{confidence_pct:.1f}%")

            # Probability Chart
            st.write("**Classification Confidence:**")
            chart_data = {LABEL_NAMES[i].capitalize(): float(probs[i]) for i in range(4)}
            st.bar_chart(chart_data)
else:
    st.info("Please upload an image to begin.")
