# app.py
"""
Streamlit: Friendly UI for Real-time Human Movement Detector
Single-file app for non-technical users: big buttons, simple instructions,
preset questions, friendly labels, clear history table.
Requirements:
pip install streamlit opencv-python mediapipe numpy pandas
Run:
streamlit run app.py
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque, Counter
from datetime import datetime, timedelta

# ---------- SETTINGS ----------
st.set_page_config(page_title="Easy Human Detector", layout="wide", initial_sidebar_state="auto")
FRAME_CONFIRM = 3
MAX_HISTORY = 500
WALK_SPEED_THRESHOLD = 0.012
EAT_DISTANCE_THRESHOLD = 0.12

# ---------- Helpers ----------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# Tracker class (keeps history & debounce)
class PersonStateTracker:
    def __init__(self, max_history=MAX_HISTORY):
        self.history = deque(maxlen=max_history)
        self.buffer = deque(maxlen=FRAME_CONFIRM)
        self.prev_hip_x = None

    def update(self, detected):
        self.buffer.append(detected)
        if len(self.buffer) == FRAME_CONFIRM:
            keys = set().union(*[d.keys() for d in self.buffer])
            confirmed = {}
            for k in keys:
                vals = [f.get(k, None) for f in self.buffer]
                confirmed[k] = Counter(vals).most_common(1)[0][0]
            entry = {"time": datetime.utcnow(), "state": confirmed}
            self.history.append(entry)
            return confirmed
        return None

    def last_n(self, seconds=10):
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [h for h in self.history if h["time"] >= cutoff]

    def last(self):
        return self.history[-1] if self.history else None

# ---------- Streamlit UI (layout) ----------
st.markdown("<h1 style='text-align:center'>🔎 Easy Human Movement Detector</h1>", unsafe_allow_html=True)
st.write("Simple interface — koi technical cheez zaroori nahi. Start camera par click karo aur screen par dekhte hi result mil jayega.")
st.write("Aasan commands: **Start camera**, **Stop camera**, aur preset questions. (Roman Urdu friendly labels.)")

left, right = st.columns((2,1))

with right:
    st.markdown("### ⚙️ Controls")
    start = st.button("▶️ Start camera", key="start", help="Click to start webcam")
    stop = st.button("⏹ Stop camera", key="stop", help="Click to stop webcam")
    st.markdown("---")
    st.markdown("### 👇 Preset Questions (click to ask)")
    q1 = st.button("Was person sitting 10s ago?", key="q1")
    q2 = st.button("Was hand up 10s ago?", key="q2")
    q3 = st.button("Is person currently walking?", key="q3")
    st.markdown("---")
    st.markdown("### ⚠️ Help / Tips")
    with st.expander("How to use (quick)"):
        st.markdown("""
- Step 1: Click **Start camera**.  
- Step 2: Stand in front of camera, simple movements like `hand up`, `sit`, `walk` will be detected.  
- Step 3: Use preset questions or type your own in the Q&A box.  
- Use **Stop camera** to end session.  
""")
    st.markdown("---")
    st.markdown("### Language")
    st.write("Labels are simple English + Roman Urdu for clarity.")

with left:
    img_placeholder = st.image(np.zeros((480,640,3), dtype=np.uint8))
    status_text = st.empty()
    labels_box = st.empty()

# Bottom area: history & custom query
st.markdown("---")
col_a, col_b = st.columns((3,1))
with col_a:
    st.markdown("### 📜 Recent Events (latest first)")
    hist_table = st.empty()
with col_b:
    st.markdown("### ❓ Ask a question")
    user_q = st.text_input("Type here (e.g., 'Was he sitting 15s ago?' or 'Hand up?')", "")
    ask = st.button("Ask", key="ask")
    clear_history = st.button("Clear history", key="clear_hist")

# ---------- App state ----------
if "running" not in st.session_state:
    st.session_state.running = False
if "tracker" not in st.session_state:
    st.session_state.tracker = PersonStateTracker()
if "cap" not in st.session_state:
    st.session_state.cap = None
if "pose" not in st.session_state:
    st.session_state.pose = None
if "hands" not in st.session_state:
    st.session_state.hands = None

# handle clear
if clear_history:
    st.session_state.tracker = PersonStateTracker()
    st.success("History cleared ✅")

# Start/Stop logic
if start and not st.session_state.running:
    st.session_state.cap = cv2.VideoCapture(0)
    if not st.session_state.cap.isOpened():
        st.error("Camera not available. Close other apps using camera or check permissions.")
    else:
        st.session_state.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        st.session_state.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        st.session_state.running = True
        st.success("Camera started — aasan mode on ✅")

if stop and st.session_state.running:
    if st.session_state.cap:
        st.session_state.cap.release()
    if st.session_state.pose:
        st.session_state.pose.close()
    if st.session_state.hands:
        st.session_state.hands.close()
    st.session_state.running = False
    st.info("Camera stopped.")

# Small function to convert history to display-friendly table
def history_to_df(hist_deque, n=50):
    rows = []
    last = list(hist_deque)[-n:][::-1]
    for e in last:
        t = e["time"].strftime("%H:%M:%S")
        s = e["state"]
        rows.append({
            "time": t,
            "posture": s.get("posture"),
            "action": s.get("action"),
            "R_hand": s.get("hand_right"),
            "L_hand": s.get("hand_left"),
            "eating": s.get("likely_eating")
        })
    if not rows:
        return pd.DataFrame(columns=["time","posture","action","R_hand","L_hand","eating"])
    return pd.DataFrame(rows)

# ---------- Main camera loop (runs while session_state.running) ----------
if st.session_state.running:
    cap = st.session_state.cap
    pose = st.session_state.pose
    hands = st.session_state.hands

    # small live loop (stop by clicking Stop camera)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("No camera frame. Stopping.")
            st.session_state.running = False
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        # default detection
        detected = {
            "posture": "unknown",
            "action": "static",
            "hand_right": "down",
            "hand_left": "down",
            "likely_eating": False
        }

        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            left_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            right_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
            left_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y])
            right_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y])
            left_knee = np.array([lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y])
            right_knee = np.array([lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y])
            left_wrist = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y])
            right_wrist = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
            nose = np.array([lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y])

            shoulder_mid = (left_sh + right_sh) / 2.0
            hip_mid = (left_hip + right_hip) / 2.0
            knee_mid = (left_knee + right_knee) / 2.0
            torso_len = np.linalg.norm(shoulder_mid - hip_mid) + 1e-8

            # posture
            if (hip_mid[1] - knee_mid[1]) < 0.06:
                detected["posture"] = "sitting"
            else:
                detected["posture"] = "standing"

            # walking (hip x velocity)
            hip_x = hip_mid[0]
            prev = st.session_state.tracker.prev_hip_x if hasattr(st.session_state.tracker, "prev_hip_x") else None
            if prev is not None:
                dx = abs(hip_x - prev)
                if dx > WALK_SPEED_THRESHOLD:
                    detected["action"] = "walking"
                else:
                    detected["action"] = "static"
            st.session_state.tracker.prev_hip_x = hip_x

            # hands relative to shoulder/hip
            sh_y = min(left_sh[1], right_sh[1])
            if right_wrist[1] < sh_y:
                detected["hand_right"] = "up"
            elif right_wrist[1] > hip_mid[1]:
                detected["hand_right"] = "down"
            else:
                detected["hand_right"] = "middle"
            if left_wrist[1] < sh_y:
                detected["hand_left"] = "up"
            elif left_wrist[1] > hip_mid[1]:
                detected["hand_left"] = "down"
            else:
                detected["hand_left"] = "middle"

            # likely eating
            dist_r = np.linalg.norm(right_wrist - nose) / torso_len
            dist_l = np.linalg.norm(left_wrist - nose) / torso_len
            if dist_r < EAT_DISTANCE_THRESHOLD or dist_l < EAT_DISTANCE_THRESHOLD:
                detected["likely_eating"] = True

        # update tracker & maybe get confirmed state
        confirmed = st.session_state.tracker.update(detected)

        # draw landmarks on frame for user
        disp = frame.copy()
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(disp, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
        if results_hands.multi_hand_landmarks:
            for h_lms in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(disp, h_lms, mp_hands.HAND_CONNECTIONS)

        # overlay friendly label (Roman Urdu + short)
        show_lines = []
        if confirmed:
            show_lines.append(f"Posture: {confirmed.get('posture','-')}  (Khara/Baitha)")
            show_lines.append(f"Action: {confirmed.get('action','-')}  (Walking/Static)")
            show_lines.append(f"Right hand: {confirmed.get('hand_right','-')}")
            show_lines.append(f"Left hand: {confirmed.get('hand_left','-')}")
            show_lines.append(f"Likely eating: {confirmed.get('likely_eating',False)}")
        else:
            last = st.session_state.tracker.last()
            if last:
                s = last["state"]
                show_lines.append(f"Last: {s.get('posture','-')}, {s.get('action','-')}")
            else:
                show_lines.append("No confirmed state yet — hold a few seconds")

        y0 = 30
        for i, ln in enumerate(show_lines):
            cv2.putText(disp, ln, (10, y0 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

        # update UI
        img_placeholder.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        status_text.info(f"Running — {now_str()}  •  Confirm buffer: {len(st.session_state.tracker.buffer)}/{FRAME_CONFIRM}")

        # show history table
        df_hist = history_to_df(st.session_state.tracker.history, n=50)
        hist_table.table(df_hist)

        # auto-answer preset question buttons
        if q1:
            recent = st.session_state.tracker.last_n(10)
            found = any(r["state"].get("posture") == "sitting" for r in recent)
            st.toast("Yes" if found else "No")
            q1 = False
        if q2:
            recent = st.session_state.tracker.last_n(10)
            found = any((r["state"].get("hand_left")=="up" or r["state"].get("hand_right")=="up") for r in recent)
            st.toast("Yes" if found else "No")
            q2 = False
        if q3:
            laststate = st.session_state.tracker.last()
            is_walk = laststate and laststate["state"].get("action") == "walking"
            st.toast("Walking now" if is_walk else "Not walking")
            q3 = False

        # custom query handling
        if ask or user_q:
            q_text = user_q.lower()
            seconds = 10
            import re
            m = re.search(r'(\d+)\s*s', q_text)
            if m:
                seconds = int(m.group(1))
            recent_list = st.session_state.tracker.last_n(seconds)
            answer = "No history to answer."
            if "sit" in q_text or "sitting" in q_text:
                found = any(r["state"].get("posture") == "sitting" for r in recent_list)
                answer = f"Sitting in last {seconds}s: {'Yes' if found else 'No'}"
            elif "stand" in q_text or "standing" in q_text:
                found = any(r["state"].get("posture") == "standing" for r in recent_list)
                answer = f"Standing in last {seconds}s: {'Yes' if found else 'No'}"
            elif "hand" in q_text or "haath" in q_text:
                found = any((r["state"].get("hand_left")=="up" or r["state"].get("hand_right")=="up") for r in recent_list)
                answer = f"Hand up in last {seconds}s: {'Yes' if found else 'No'}"
            elif "eat" in q_text or "khana" in q_text:
                found = any(r["state"].get("likely_eating") for r in recent_list)
                answer = f"Eating (likely) in last {seconds}s: {'Yes' if found else 'No'}"
            else:
                last = st.session_state.tracker.last()
                if last:
                    s = last["state"]
                    answer = f"Last: posture={s.get('posture')}, action={s.get('action')}, Rhand={s.get('hand_right')}"
            st.info(answer)
            ask = False
            user_q = ""

        # small sleep to be gentle on CPU & approx FPS
        time.sleep(0.05)

    # cleanup after loop ends
    if st.session_state.cap:
        st.session_state.cap.release()
    if st.session_state.pose:
        st.session_state.pose.close()
    if st.session_state.hands:
        st.session_state.hands.close()
    st.session_state.running = False
    status_text.warning("Camera stopped.")

else:
    status_text.info("Camera is stopped. Click 'Start camera' to begin.")
    df_hist = history_to_df(st.session_state.tracker.history, n=20)
    hist_table.table(df_hist)
