import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Medical/Clean Theme) ---
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00897b; /* Teal for medical feel */
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00695c;
        color: white;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        color: #00897b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2813/2813292.png", width=70) # Mask Icon
with col2:
    st.title("HealthGuard Compliance System")
    st.markdown("**Automated Face Mask Screening** | Powered by YOLOv11")

st.markdown("---")

# --- Sidebar Settings ---
st.sidebar.title("⚙️ System Config")
st.sidebar.subheader("Screening Sensitivity")

# Confidence Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.50, 
    step=0.01,
    help="Adjust sensitivity. Higher values minimize false detections."
)

# App Mode Selector
app_mode = st.sidebar.selectbox(
    "Select Operation Mode",
    ["Protocol Audit (Image)", "Walkway Monitor (Video)", "Entry Screening (Live)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Ensure your trained 'best.pt' (Mask Model) is in the root directory.")

# --- Model Loading (Absolute Path Fix) ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"⚠️ Model file not found at: {model_path}")
    st.info("Please rename your trained mask model to 'best.pt' and place it here.")
    model = None

# --- Main Logic ---

if model:
    # ---------------- IMAGE MODE (Audit) ----------------
    if app_mode == "Protocol Audit (Image)":
        st.subheader("📸 Compliance Audit")
        uploaded_file = st.file_uploader("Upload personnel image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Photo", use_container_width=True)

            with col2:
                if st.button("Check Compliance", type="primary"):
                    with st.spinner("Scanning faces..."):
                        results = model(img_array, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        
                        st.image(annotated_img, caption="Screening Result", use_container_width=True)
                        
                        # Metrics
                        count = len(results[0].boxes)
                        st.metric(label="Faces Detected", value=count)
                        
                        if count > 0:
                            st.success("Screening Complete.")
                        else:
                            st.warning("No faces/masks detected.")

    # ---------------- VIDEO MODE (Walkway) ----------------
    elif app_mode == "Walkway Monitor (Video)":
        st.subheader("🎥 Walkway Surveillance")
        video_file = st.file_uploader("Upload surveillance footage", type=["mp4", "avi", "mov"])

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            st.sidebar.markdown("---")
            stop_button = st.sidebar.button("Stop Monitoring")

            col1, col2 = st.columns([3, 1])
            with col1:
                stframe = st.empty()
            with col2:
                st.markdown("### 📊 Live Data")
                kpi_text = st.empty()

            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
                    break

                results = model(frame, conf=conf_threshold)
                annotated = results[0].plot()
                
                stframe.image(annotated, channels="BGR", use_container_width=True)
                
                # Counter
                count = len(results[0].boxes)
                kpi_text.markdown(f"**Current Count:** {count}")

            cap.release()
            st.success("Monitoring session ended.")

    # ---------------- LIVE MODE (Entry Screening) ----------------
    elif app_mode == "Entry Screening (Live)":
        st.subheader("🔴 Live Entry Checkpoint")
        st.write("Real-time screening for building entry.")

        run = st.checkbox('Activate Scanner', value=False)
        
        frame_window = st.image([])
        cap = cv2.VideoCapture(0)

        if run:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera disconnected.")
                    break
                
                results = model(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for Streamlit
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                frame_window.image(annotated_frame)
        else:
            cap.release()
            st.write("Scanner is in standby mode.")