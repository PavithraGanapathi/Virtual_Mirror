"""
The Silk Foundry - Virtual Mirror Saree Draping POC (Starter Kit)
Architect: Mayan (CTO Assistant) & Candidate Challenge Baseline

Usage:
    pip install -r requirements.txt
    streamlit run starter_virtual_mirror.py
"""
import os
import math
import numpy as np
from PIL import Image

try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class VirtualMirrorEngine:
    """
    Core computer vision engine for pose landmark extraction,
    garment affine transformation, and alpha blending.
    """
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, image_rgb: np.ndarray):
        """
        Extracts key upper body landmarks:
        - 11: Left Shoulder
        - 12: Right Shoulder
        - 23: Left Hip
        - 24: Right Hip
        """
        if not MEDIAPIPE_AVAILABLE:
            return None, "MediaPipe not installed. Please run: pip install mediapipe"

        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, "No human pose detected in the image."

        h, w, _ = image_rgb.shape
        lm = results.pose_landmarks.landmark

        landmarks_dict = {
            "left_shoulder": (int(lm[11].x * w), int(lm[11].y * h)),
            "right_shoulder": (int(lm[12].x * w), int(lm[12].y * h)),
            "left_hip": (int(lm[23].x * w), int(lm[23].y * h)),
            "right_hip": (int(lm[24].x * w), int(lm[24].y * h)),
            "nose": (int(lm[0].x * w), int(lm[0].y * h))
        }

        # Calculate geometric metrics
        ls = landmarks_dict["left_shoulder"]
        rs = landmarks_dict["right_shoulder"]
        lh = landmarks_dict["left_hip"]
        rh = landmarks_dict["right_hip"]

        # Shoulder width in pixels
                # Shoulder and hip geometry
        shoulder_width = math.hypot(
            ls[0] - rs[0],
            ls[1] - rs[1]
        )

        hip_width = math.hypot(
            lh[0] - rh[0],
            lh[1] - rh[1]
        )

        # Angle of the person's anatomical left-to-right shoulder line.
        # For this sample, this is approximately -0.63 degrees.
        shoulder_angle_rad = math.atan2(
            ls[1] - rs[1],
            ls[0] - rs[0]
        )
        shoulder_angle_deg = math.degrees(shoulder_angle_rad)

        # Torso geometry from the midpoints.
        shoulder_mid = (
            (ls[0] + rs[0]) // 2,
            (ls[1] + rs[1]) // 2
        )
        hip_mid = (
            (lh[0] + rh[0]) // 2,
            (lh[1] + rh[1]) // 2
        )
        torso_height = math.hypot(
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1]
        )

        metrics = {
            "shoulder_width_px": round(shoulder_width, 1),
            "hip_width_px": round(hip_width, 1),
            "shoulder_tilt_deg": round(shoulder_angle_deg, 2),
            "torso_height_px": round(torso_height, 1),
            "shoulder_mid": shoulder_mid,
            "hip_mid": hip_mid
        }

        return {"landmarks": landmarks_dict, "metrics": metrics, "raw_results": results}, None

        # Location of the customer shoulder inside the pre-draped PNG,
    # expressed as fractions of the resized garment image.
    SAREE_SHOULDER_ANCHOR_X = 0.55
    SAREE_SHOULDER_ANCHOR_Y = 0.15

    
    def drape_saree(
        self,
        customer_pil: Image.Image,
        saree_pil: Image.Image,
        landmarks_data: dict,
        scale_mult: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0
    ) -> Image.Image:
        """
        Rotates, scales, and overlays transparent saree PNG onto customer photo
        aligned to left shoulder and torso landmarks.
        """
        metrics = landmarks_data["metrics"]
        lm = landmarks_data["landmarks"]

        # Preserve the pre-draped PNG's natural proportions.
        target_width = int(
            metrics["shoulder_width_px"] * 2.2 * scale_mult
        )

        orig_w, orig_h = saree_pil.size
        aspect = orig_h / float(orig_w)
        target_height = int(target_width * aspect)

        resized_saree = saree_pil.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )

        rotated_saree = resized_saree.rotate(
            -metrics["shoulder_tilt_deg"],
            expand=True,
            resample=Image.Resampling.BICUBIC
        )

        # MediaPipe "left shoulder" means the customer's anatomical left:
        # on a front-facing photo, it appears on the viewer's right.
        shoulder = lm["left_shoulder"]

        # Place the embedded garment shoulder anchor over the detected shoulder.
        paste_x = (
            shoulder[0]
            - int(target_width * self.SAREE_SHOULDER_ANCHOR_X)
            + offset_x
        )
        paste_y = (
            shoulder[1]
            - int(target_height * self.SAREE_SHOULDER_ANCHOR_Y)
            + offset_y
        )

        # Prepare composite image
        output = customer_pil.convert("RGBA")

        # Apply opacity adjustment
        if opacity < 1.0:
            r, g, b, a = rotated_saree.split()
            a = a.point(lambda p: int(p * opacity))
            rotated_saree = Image.merge("RGBA", (r, g, b, a))

        output.paste(rotated_saree, (paste_x, paste_y), mask=rotated_saree)
        return output.convert("RGB")


# =====================================================================
# Streamlit Interactive Interface
# =====================================================================
def run_app():
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="The Silk Foundry - Virtual Mirror POC", layout="wide")

    st.title("🪞 The Silk Foundry &bull; Saree Virtual Mirror POC")
    st.markdown("Automated pose estimation, landmark tracking, and 2D garment drape alignment.")

    engine = VirtualMirrorEngine()

    # Sidebar: Asset & Garment Selection
    st.sidebar.header("1. Garment Catalog")
    saree_choice = st.sidebar.selectbox(
        "Select Saree Design",
        ["Kanchipuram Crimson & Gold", "Royal Peacock Blue & Gold"]
    )

    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    saree_path = os.path.join(asset_dir, "saree_crimson.png" if "Crimson" in saree_choice else "saree_blue.png")

    if os.path.exists(saree_path):
        saree_img = Image.open(saree_path).convert("RGBA")
        st.sidebar.image(saree_img, caption=saree_choice, use_container_width=True)
    else:
        st.sidebar.error(f"Saree asset missing: {saree_path}")
        return

    # Sidebar: Input Source Selection
    st.sidebar.header("2. Customer Input Source")
    input_source = st.sidebar.radio("Input Type", ["Sample Model Photo", "Upload Photo", "Webcam Capture"])

    default_customer_path = os.path.join(asset_dir, "sample_customer.jpg")
    customer_img = None

    if input_source == "Sample Model Photo":
        if os.path.exists(default_customer_path):
            customer_img = Image.open(default_customer_path)
        else:
            st.error("Default sample model photo missing.")
    elif input_source == "Upload Photo":
        uploaded_file = st.sidebar.file_uploader("Upload full upper-body photo", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            customer_img = Image.open(uploaded_file)
    elif input_source == "Webcam Capture":
        camera_file = st.sidebar.camera_input("Capture Pose Photo")
        if camera_file:
            customer_img = Image.open(camera_file)

    if customer_img is None:
        st.info("Please select or upload a customer photo to proceed.")
        return

    # Sidebar: Calibration Controls
    st.sidebar.header("3. Draping Calibration")
    scale_slider = st.sidebar.slider("Saree Scale Multiplier", 0.7, 1.6, 0.95 , 0.05)
    offset_x = st.sidebar.slider("Horizontal Offset (X)", -150, 150, -60, 5)
    offset_y = st.sidebar.slider("Vertical Offset (Y)", -150, 150, 15, 5)
    opacity_slider = st.sidebar.slider("Garment Opacity", 0.5, 1.0, 1.0, 0.05)
    show_landmarks = st.sidebar.checkbox("Overlay Pose Landmarks Skeleton", value=False)

    # Core Pipeline Execution
    customer_np = np.array(customer_img.convert("RGB"))
    detection_res, err = engine.extract_landmarks(customer_np)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Input Customer Photo")
        if detection_res and show_landmarks:
            debug_img = customer_np.copy()
            engine.mp_drawing.draw_landmarks(
                debug_img,
                detection_res["raw_results"].pose_landmarks,
                engine.mp_pose.POSE_CONNECTIONS
            )
            st.image(debug_img, caption="Pose Landmarks Extracted", use_container_width=True)
        else:
            st.image(customer_img, caption="Original Input", use_container_width=True)

    with col2:
        st.subheader("✨ Virtual Mirror Draping Output")
        if err:
            st.warning(f"Detection warning: {err}")
            st.image(customer_img, use_container_width=True)
        else:
            draped_result = engine.drape_saree(
                customer_pil=customer_img,
                saree_pil=saree_img,
                landmarks_data=detection_res,
                scale_mult=scale_slider,
                offset_x=offset_x,
                offset_y=offset_y,
                opacity=opacity_slider
            )
            st.image(draped_result, caption="Real-time Saree Draping Alignment", use_container_width=True)

            # Display Telemetry
            m = detection_res["metrics"]
            st.markdown(f"""
            **Pose Telemetry:**
            - **Shoulder Width:** `{m['shoulder_width_px']} px`
            - **Hip Width:** `{m['hip_width_px']} px`
            - **Torso Height:** `{m['torso_height_px']} px`
            - **Shoulder Tilt Angle:** `{m['shoulder_tilt_deg']}°`
            """)


if __name__ == "__main__":
    run_app()
