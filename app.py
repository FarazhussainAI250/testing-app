import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Movement Detector Demo", layout="wide")

st.markdown("<h1 style='text-align:center'>🔎 Human Movement Detection Demo</h1>", unsafe_allow_html=True)
st.write("Demo version - MediaPipe/OpenCV dependencies removed for cloud deployment")

col1, col2 = st.columns([3,1])

with col2:
    st.markdown("### Controls")
    start_demo = st.button("▶️ Start Demo", key="start")
    stop_demo = st.button("⏹ Stop Demo", key="stop")
    
    st.markdown("### Sample Detection")
    sample_data = {
        "posture": "sitting",
        "action": "static", 
        "hand_right": "up",
        "hand_left": "down",
        "likely_eating": False
    }
    
    for key, value in sample_data.items():
        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

with col1:
    if "demo_running" not in st.session_state:
        st.session_state.demo_running = False
    
    if start_demo:
        st.session_state.demo_running = True
    
    if stop_demo:
        st.session_state.demo_running = False
    
    if st.session_state.demo_running:
        st.info("Demo running — browser will ask for camera permission. Take a snapshot for processing below.")
        
        # Camera input (single snapshot). Works well on Streamlit Cloud.
        img_file = st.camera_input("Take a snapshot")

        if img_file:
            # Display captured image
            st.image(img_file, caption="Captured frame", use_column_width=True)
            
            # Convert to numpy array for any processing you want to do
            img = Image.open(io.BytesIO(img_file.getvalue()))
            arr = np.array(img)
            st.write("Captured image shape:", arr.shape)

            # ----
            # Place your detection logic here using `arr`
            # e.g., run a model/pipeline and then show results
            # For demo, we'll show random "detection" result
            demo_results = {
                "posture": "sitting",
                "action": "static",
                "hand_right": "up",
                "hand_left": "down",
                "likely_eating": False
            }
            st.success("Detection results (demo):")
            for k, v in demo_results.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v}")
            # ----
        else:
            st.write("Click 'Take a snapshot' to capture a frame.")
    else:
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        st.image(placeholder_img, caption="Click Start Demo to begin")

st.markdown("---")
st.markdown("### Features")
st.markdown("""
- **Posture Detection:** Sitting vs Standing
- **Action Recognition:** Walking vs Static  
- **Hand Position:** Up, Down, Middle
- **Eating Detection:** Hand-to-face proximity
- **Real-time Analysis:** Frame processing (use streamlit-webrtc for true live)
""")

st.info("This is a demo version. Full version includes MediaPipe pose detection and live camera integration.")
