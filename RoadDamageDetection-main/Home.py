import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection Platform",
    page_icon="🛣️",
)

st.divider()
st.title("🛣️ Road Damage Detection & Classification Platform")

st.markdown(
    """
    ## Welcome to Advanced Road Infrastructure Analysis
    
    Our comprehensive platform combines cutting-edge AI technologies to revolutionize road maintenance and infrastructure monitoring. 
    Using dual deep learning approaches, we provide both precise damage detection and comprehensive damage classification.

    ### 🤖 Dual AI-Powered Analysis

    #### 🎯 **Object Detection with YOLOv12**
    - **Precise Location Identification**: Detects and pinpoints exact locations of road damage with bounding boxes
    - **Real-time Processing**: Fast inference for immediate damage assessment
    - **Multiple Damage Types**: Identifies Alligator Cracking, Linear Cracking, Potholes, Patching, and Rutting
    - **Confidence Scoring**: Provides reliability scores for each detection
    - **Visual Annotations**: Overlays detection results directly on images

    #### 🧠 **Image Classification with ResNet50**
    - **Holistic Analysis**: Classifies overall damage patterns across entire road surfaces
    - **Multi-label Prediction**: Can identify multiple damage types simultaneously
    - **Transfer Learning**: Built on pre-trained ImageNet weights with fine-tuning for road damage
    - **Probability Distribution**: Shows likelihood scores for all damage categories
    - **Advanced Architecture**: Deep residual networks for robust feature extraction

    ### 🔧 **Key Features & Capabilities**

    #### **Smart Model Management**
    - ⚡ **Automated Downloads**: Models download automatically on first use
    - 🔄 **Intelligent Caching**: No repeated downloads, faster subsequent loading
    - 💾 **Local Storage**: Models stored locally for offline usage
    - 🛡️ **Error Handling**: Graceful fallbacks and user-friendly error messages

    #### **Professional Interface**
    - 📱 **Responsive Design**: Works seamlessly across devices
    - 🎨 **Clean UI/UX**: Intuitive interface with professional styling
    - 📊 **Rich Visualizations**: Charts, metrics, and detailed result displays
    - 📥 **Export Functionality**: Download annotated images and results

    ### 🎯 **Damage Categories Detected**

    | Damage Type | Description | Detection Method |
    |-------------|-------------|------------------|
    | **Linear Cracking** | Linear fractures in road surface | YOLO + ResNet |
    | **Alligator Cracking** | Interconnected crack patterns resembling alligator skin | YOLO + ResNet |
    | **Pothole** | Circular holes or depressions in pavement | YOLO + ResNet |
    | **Patching** | Previously repaired areas with different materials | YOLO + ResNet |
    | **Rutting** | Longitudinal depressions caused by wheel loading | YOLO + ResNet |

    ### 🚀 **Getting Started**

    1. **Choose Your Analysis Method**:
       - Use **Road damage detection - YOLOv12** for precise location-based detection
       - Use **Road damage classification - ResNet** for overall damage assessment

    2. **Upload Your Image**: Support for PNG, JPG, and JPEG formats

    3. **Adjust Settings**: Fine-tune confidence thresholds for optimal results

    4. **Analyze Results**: View detailed detection/classification results with confidence scores

    5. **Export & Share**: Download annotated images for reports and documentation

    ### 💡 **Use Cases**

    - **Municipal Maintenance**: Prioritize road repair schedules based on damage severity
    - **Infrastructure Audits**: Comprehensive assessment of road network conditions
    - **Construction Quality Control**: Verify road surface quality post-construction
    - **Insurance Claims**: Document road damage for insurance and liability purposes
    - **Research & Development**: Analyze road deterioration patterns and material performance

    ### 🔬 **Technical Specifications**

    - **YOLO Model**: YOLOv12 Nano - Latest architecture with improved accuracy
    - **Training Details**: 300 epochs, 3,386 training images, 640x640 resolution
    - **Dataset Split**: 70% train / 20% validation / 10% test (4,844 total images)
    - **ResNet Model**: ResNet50 with transfer learning - Deep feature extraction
    - **Input Resolution**: 640x640 (YOLO), 224x224 (ResNet)
    - **Inference Device**: Automatic GPU/CPU detection for optimal performance
    - **Supported Formats**: PNG, JPG, JPEG image files

    ---

    ### 📚 **Documentation & Resources**
    - 🌐 **Project Repository**: [GitHub - RoadDamageDetection](https://github.com/achilis1505/RoadDamageDetection)
    - 📧 **Contact**: tasmiayemna@gmail.com
    - 📖 **Framework Credits**: Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/)

    ### ⚖️ **License & Attribution**
    - YOLOv8: Licensed under AGPL-3.0 by [Ultralytics](https://github.com/ultralytics/ultralytics)
    - Streamlit: Licensed under Apache 2.0 by [Streamlit Inc.](https://streamlit.io/)
    - ResNet: PyTorch implementation with ImageNet pre-trained weights

    ---

    **Ready to analyze road conditions?** Select a detection method from the sidebar to begin! 🚀
    """
)
