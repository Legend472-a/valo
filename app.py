import os
import torch
torch.classes.__path__ = []  # Fix for Streamlit + Torch

import streamlit as st
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# --------------------- Config ---------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------------------- Model ---------------------
class TransferLearningResNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

@st.cache_resource
def load_model():
    model = TransferLearningResNet(num_classes=4).to(device)
    model_path = "resnet_model.pth"
    
    if not os.path.exists(model_path):
        st.error("❌ resnet_model.pth not found! Please place it in the same folder.")
        st.stop()
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# --------------------- Transform ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------------------- Main UI ---------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("**Upload a brain MRI image to get instant classification**")

# Upload Section
uploaded_file = st.file_uploader(
    "Choose MRI Image (JPG / PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Recommended: Clear axial view MRI"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Two columns for better space
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        with st.spinner("Analyzing..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                confidence, pred_idx = torch.max(probs, 0)
            
            predicted_class = LABEL_NAMES[pred_idx.item()]
            confidence_pct = confidence.item() * 100

        # Big Result
        if predicted_class == "notumor":
            st.success("✅ **NO TUMOR DETECTED**", icon="🟢")
        else:
            st.error(f"⚠️ **{predicted_class.upper()} TUMOR** detected", icon="🔴")
        
        st.metric("Confidence", f"{confidence_pct:.1f}%")
        st.progress(confidence.item())

        # All Probabilities
        st.subheader("Class Probabilities")
        prob_dict = {LABEL_NAMES[i].capitalize(): float(probs[i])*100 for i in range(4)}
        st.bar_chart(prob_dict, use_container_width=True)
