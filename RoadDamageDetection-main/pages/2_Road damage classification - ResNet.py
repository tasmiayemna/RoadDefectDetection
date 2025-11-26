# pages/resnet.py
import os
import logging
from pathlib import Path
from typing import NamedTuple
import io

import numpy as np
import streamlit as st

# Deep learning framework
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Road damage classification - ResNet",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Single place to set your local model filename
MODEL_DIR = ROOT / "models"
MODEL_FILENAME = "resnet.pth"   # change if your filename differs
MODEL_LOCAL_PATH = MODEL_DIR / MODEL_FILENAME

MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ImageClassifier:
    def __init__(self, model_path=None, num_classes=5):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"Initialized ImageClassifier with device: {self.device}")

    def load_model(self):
        if self.model is not None:
            return

        logger.info("Initializing ResNet50 with ImageNet weights")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        logger.info("Freezing feature extraction layers")
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.Sigmoid()
        )

        if self.model_path and os.path.exists(self.model_path):
            try:
                logger.info(f"Attempting to load model weights from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    raise ValueError("Unexpected checkpoint format")

                # strip 'module.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[len("module."):] if k.startswith("module.") else k
                    new_state_dict[new_key] = v

                try:
                    self.model.load_state_dict(new_state_dict, strict=True)
                    logger.info("Loaded model weights with strict=True")
                except Exception as e_strict:
                    logger.warning(f"Strict load failed: {e_strict}. Trying non-strict load.")
                    self.model.load_state_dict(new_state_dict, strict=False)
                    logger.info("Loaded model weights with strict=False")

            except Exception as e:
                logger.error(f"Failed to load model weights: {str(e)}")
                logger.warning("Continuing with ImageNet weights only")
        else:
            logger.warning("No local model file found. Using ImageNet weights only")

        self.model.to(self.device)
        self.model.eval()
        logger.info("Model initialized and moved to device")

    def predict(self, image_input, threshold=0.5):
        if self.model is None:
            self.load_model()

        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Input must be PIL Image or bytes")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (probabilities > threshold).astype(float)

        class_names = ['Rutting', 'Crack', 'Patch', 'Pothole', 'Raveling']
        predicted_classes = [class_names[i] for i, pred in enumerate(predictions) if pred == 1]

        if not predicted_classes:
            max_prob_idx = int(np.argmax(probabilities))
            predicted_classes = [class_names[max_prob_idx]]

        all_probabilities = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

        confidence = float(max(probabilities)) if len(predicted_classes) == 1 else float(np.mean([probabilities[class_names.index(cls)] for cls in predicted_classes]))

        return {
            "predicted_classes": predicted_classes,
            "primary_class": predicted_classes[0] if predicted_classes else "Unknown",
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "raw_probabilities": probabilities.tolist(),
            "model_loaded": self.model_path is not None and os.path.exists(self.model_path) if self.model_path else False
        }

resnet_classifier = ImageClassifier(model_path=str(MODEL_LOCAL_PATH), num_classes=5)
with st.spinner("Loading ResNet model..."):
    resnet_classifier.load_model()

ROAD_DAMAGE_CLASSES = [
    "Rutting",
    "Crack",
    "Patch",
    "Pothole",
    "Raveling"
]

st.title("Road Damage Classification - ResNet")

st.write("""
Classify road damage types using a ResNet deep learning model. This approach uses image classification 
to categorize the entire image into different damage types. The model can predict multiple damage types 
simultaneously using multi-label classification.
""")

st.write("Upload an image to classify the type of road damage present.")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of a road to classify damage type"
)

confidence_threshold = st.slider(
    "Classification Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Minimum probability score to consider a class as detected"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    try:
        # robust display for different Streamlit versions
        try:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except TypeError:
            st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception:
        # fallback plain display
        st.image(image)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Width", f"{image.size[0]}px")
    with col2:
        st.metric("Height", f"{image.size[1]}px")
    with col3:
        st.metric("Mode", image.mode)

    st.divider()

    st.subheader("Classification Results")

    with st.spinner("Classifying image using ResNet..."):
        try:
            prediction = resnet_classifier.predict(image, threshold=confidence_threshold)
            predicted_classes = prediction["predicted_classes"]
            primary_class = prediction["primary_class"]
            confidence = prediction["confidence"]
            all_probs = prediction["all_probabilities"]
            raw_probs = prediction["raw_probabilities"]
            model_loaded = prediction.get("model_loaded", False)

            if model_loaded:
                st.success("🎯 Using fine-tuned ResNet model")
            else:
                st.warning("⚠️ Using ImageNet weights only (not trained for road damage)")

            if predicted_classes:
                detected_names = ", ".join(predicted_classes)
                st.success(f"**Detected Classes:** {detected_names}")
            else:
                st.warning("No damage classes detected above threshold")
                max_prob_class = max(all_probs.items(), key=lambda x: x[1])
                st.write(f"**Highest Probability:** {max_prob_class[0]} ({max_prob_class[1]:.2%})")

            st.subheader("All Class Probabilities")
            prob_data = []
            for class_name, prob in all_probs.items():
                status = "✅ Detected" if prob >= confidence_threshold else "❌ Below threshold"
                prob_data.append({
                    "Class": class_name,
                    "Probability": f"{prob:.2%}",
                    "Status": status,
                    "Score": prob
                })

            prob_data.sort(key=lambda x: x["Score"], reverse=True)
            st.table([{k: v for k, v in item.items() if k != "Score"} for item in prob_data])

            st.subheader("Probability Distribution")
            chart_data = {item["Class"]: item["Score"] for item in prob_data}
            st.bar_chart(chart_data)

            device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            st.info(f"**Inference Device:** {device_info}")

        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
            logger.error(f"Classification error: {e}")
else:
    st.info("👆 Please upload an image to start classification")
