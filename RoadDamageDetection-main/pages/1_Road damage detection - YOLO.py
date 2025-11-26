# pages/yolo.py
import logging
from pathlib import Path
from typing import NamedTuple
import numpy as np
import streamlit as st
from PIL import Image
import cv2

# ultralytics
from ultralytics import YOLO

st.set_page_config(
    page_title="Road damage detection - YOLO",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Single place to set your local YOLO model filename
MODEL_DIR = ROOT / "models"
MODEL_FILENAME = "yolov12.pt"  # <-- change to your actual filename if different
MODEL_LOCAL_PATH = MODEL_DIR / MODEL_FILENAME

# Ensure models folder exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Unified class names (use same order as ResNet mapping if you want consistency)
CLASSES = [
    "Crack",
    "Alligator Crack",
    "Pothole",
    "Patch",
    "Rutting"
]


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


@st.cache_resource
def load_yolo_model(path: str):
    try:
        if not Path(path).exists():
            logger.warning(f"YOLO model not found at {path}")
            return None
        # Loading model via ultralytics
        model = YOLO(path)
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None


with st.spinner("Loading YOLO model..."):
    net = load_yolo_model(str(MODEL_LOCAL_PATH))

st.title("Road Damage Detection - YOLO")
st.write("""
Detect road damage using a YOLO-based detector. Upload an image to detect and localize damage types.
""")

image_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of a road to detect damage"
)

score_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.01,
    help="Lower the threshold if no damage is detected, increase if there are false predictions"
)

if image_file is None:
    st.info("👆 Upload an image to start detection")
else:
    if net is None:
        st.error(f"YOLO model not loaded. Make sure models/{MODEL_FILENAME} exists in the repo and redeploy.")
    else:
        try:
            image = Image.open(image_file)
            if image.mode != "RGB":
                image = image.convert("RGB")

            st.subheader("Original Image")
            # robust display for different Streamlit versions
            try:
                st.image(image, use_container_width=True)
            except TypeError:
                st.image(image, use_column_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", f"{image.size[0]}px")
            with col2:
                st.metric("Height", f"{image.size[1]}px")
            with col3:
                st.metric("Mode", image.mode)

            st.divider()
            st.subheader("Detection Results")

            # prepare image for model, prefer letting ultralytics accept PIL/ndarray
            img_np = np.array(image)

            with st.spinner("Detecting road damage..."):
                # ultralytics model accepts images directly; pass conf param
                results = net.predict(img_np, conf=score_threshold, verbose=False)

                detections = []
                for r in results:
                    # r.boxes is a Boxes object; convert safely
                    try:
                        boxes = r.boxes.cpu().numpy()
                    except Exception:
                        # fallback: use r.boxes.xyxy, r.boxes.conf, r.boxes.cls
                        boxes = []
                        if hasattr(r.boxes, "xyxy"):
                            xyxy = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            cls = r.boxes.cls.cpu().numpy()
                            for i in range(len(xyxy)):
                                class_id = int(cls[i])
                                conf = float(confs[i])
                                box = xyxy[i].astype(int)
                                boxes.append(
                                    NamedTuple("Tmp", [("cls", int), ("conf", float), ("xyxy", np.ndarray)])(
                                        class_id, conf, box
                                    )
                                )

                    # parse boxes
                    for b in boxes:
                        # support for ultralytics vX shape
                        try:
                            class_id = int(getattr(b, "cls", b[0]))
                            conf = float(getattr(b, "conf", b[1]))
                            box = getattr(b, "xyxy", b[2]) if hasattr(b, "xyxy") else np.array(b[2]).astype(int)
                        except Exception:
                            # try index style
                            vals = list(b)
                            if len(vals) >= 4:
                                class_id = int(vals[0])
                                conf = float(vals[1])
                                box = np.array(vals[-1]).astype(int)
                            else:
                                continue

                        label = CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}"
                        detections.append(Detection(class_id=class_id, label=label, score=conf, box=box))

                # annotated image from model
                try:
                    annotated = results[0].plot()
                    # ultralytics returns numpy array in RGB
                    annotated = cv2.resize(annotated, (image.size[0], image.size[1]), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    logger.warning(f"Could not get annotated image from results: {e}")
                    annotated = img_np

                # display summary
                if detections:
                    st.success(f"🎯 **{len(detections)} damage(s) detected**")
                    damage_counts = {}
                    for d in detections:
                        damage_counts[d.label] = damage_counts.get(d.label, 0) + 1

                    damage_list = "\n\n".join([f"• ***{k}: {v} instance(s)***" for k, v in damage_counts.items()])
                    st.success(f"**Detected Damage Types:**\n\n{damage_list}")

                    with st.expander("📋 Detailed Detection Results"):
                        for i, d in enumerate(detections, 1):
                            st.write(f"**Detection {i}:**")
                            st.write(f"- Type: {d.label}")
                            st.write(f"- Confidence: {d.score:.2%}")
                            st.write(f"- Bounding Box: {d.box}")
                            st.divider()
                else:
                    st.warning("⚠️ No detections above the confidence threshold. Try lowering the threshold.")

                st.subheader("Annotated Image")
                # robust display for different Streamlit versions
                try:
                    st.image(annotated, use_container_width=True)
                except TypeError:
                    st.image(annotated, use_column_width=True)

                # Download annotated image
                from io import BytesIO

                im_pil = Image.fromarray(annotated)
                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                buf_bytes = buf.getvalue()

                st.download_button("📥 Download Prediction Image", data=buf_bytes, file_name="yolo_prediction.png",
                                   mime="image/png")

                device_info = "GPU (CUDA)" if hasattr(net, "device") and "cuda" in str(net.device) else "CPU"
                st.info(f"**Inference Device:** {device_info}")

        except Exception as e:
            st.error(f"Error during detection: {e}")
            logger.exception(e)
