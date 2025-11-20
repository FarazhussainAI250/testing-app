import streamlit as st
import numpy as np
import time
from datetime import datetime

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
        # Show demo image
        demo_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        st.image(demo_img, caption="Demo Detection Frame")
        st.success("Demo mode running!")
        
        # Auto refresh every 2 seconds
        time.sleep(2)
        st.rerun()
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
- **Real-time Analysis:** Frame processing
""")

st.info("This is a demo version. Full version includes MediaPipe pose detection and camera integration.")
