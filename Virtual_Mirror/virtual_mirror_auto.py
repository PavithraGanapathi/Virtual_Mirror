"""
Virtual Mirror Saree Draping POC — pose-relative automatic calibration
with diagnostic telemetry.

Run from this project folder:
    streamlit run virtual_mirror_pose_diagnostics.py

This file is independent of starter_virtual_mirror.py. It preserves the
pre-draped saree PNG's aspect ratio and uses an anatomical shoulder anchor.
"""

import math
import os

import numpy as np
from PIL import Image

try:
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
    """Detect pose metrics, derive automatic calibration, and composite a saree."""

    # Internal location of the intended shoulder point in the pre-draped PNG.
    SAREE_SHOULDER_ANCHOR_X = 0.55
    SAREE_SHOULDER_ANCHOR_Y = 0.15

    # Calibrated from the supplied sample image:
    # 0.95 scale, -60 px X for a 272 px shoulder span, +15 px Y for a 394 px torso.
    BASE_SCALE_MULTIPLIER = 0.95
    X_OFFSET_PER_SHOULDER_WIDTH = -60 / 272
    Y_OFFSET_PER_TORSO_HEIGHT = 15 / 394

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install -r requirements.txt")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, image_rgb: np.ndarray):
        """Return pixel landmarks and body metrics, or an explanatory error."""
        if image_rgb is None or image_rgb.size == 0:
            return None, "The selected image is empty."

        result = self.pose.process(image_rgb)
        if not result.pose_landmarks:
            return None, "No human pose detected. Try a clear, front-facing upper-body image."

        frame_height, frame_width = image_rgb.shape[:2]
        lm = result.pose_landmarks.landmark

        def pixel_point(index: int) -> tuple[int, int]:
            return int(lm[index].x * frame_width), int(lm[index].y * frame_height)

        landmarks = {
            "left_shoulder": pixel_point(11),
            "right_shoulder": pixel_point(12),
            "left_hip": pixel_point(23),
            "right_hip": pixel_point(24),
            "nose": pixel_point(0),
        }

        left_shoulder = landmarks["left_shoulder"]
        right_shoulder = landmarks["right_shoulder"]
        left_hip = landmarks["left_hip"]
        right_hip = landmarks["right_hip"]

        shoulder_width = math.hypot(
            left_shoulder[0] - right_shoulder[0],
            left_shoulder[1] - right_shoulder[1],
        )
        hip_width = math.hypot(
            left_hip[0] - right_hip[0],
            left_hip[1] - right_hip[1],
        )
        shoulder_tilt_deg = math.degrees(math.atan2(
            left_shoulder[1] - right_shoulder[1],
            left_shoulder[0] - right_shoulder[0],
        ))

        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2,
        )
        hip_mid = (
            (left_hip[0] + right_hip[0]) // 2,
            (left_hip[1] + right_hip[1]) // 2,
        )
        torso_height = math.hypot(
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1],
        )

        metrics = {
            "canvas_width": frame_width,
            "canvas_height": frame_height,
            "shoulder_width_px": round(shoulder_width, 1),
            "hip_width_px": round(hip_width, 1),
            "torso_height_px": round(torso_height, 1),
            "shoulder_tilt_deg": round(shoulder_tilt_deg, 2),
            "shoulder_mid": shoulder_mid,
            "hip_mid": hip_mid,
            "hip_to_shoulder_ratio": round(hip_width / shoulder_width, 3)
            if shoulder_width > 0 else 0.0,
            "torso_to_shoulder_ratio": round(torso_height / shoulder_width, 3)
            if shoulder_width > 0 else 0.0,
        }
        return {"landmarks": landmarks, "metrics": metrics, "raw_results": result}, None

    def automatic_calibration(
        self,
        metrics: dict,
        micro_scale: float = 1.0,
        micro_x: int = 0,
        micro_y: int = 0,
    ) -> dict:
        """Derive calibration values from pose size, not frame position.

        The landmark coordinates themselves move the garment when a person
        stands elsewhere in the image. X/Y offsets are only pre-draped asset
        alignment corrections, expressed as fractions of body dimensions.
        """
        shoulder_width = metrics["shoulder_width_px"]
        torso_height = metrics["torso_height_px"]
        return {
            "scale_mult": self.BASE_SCALE_MULTIPLIER * micro_scale,
            "offset_x": round(self.X_OFFSET_PER_SHOULDER_WIDTH * shoulder_width) + micro_x,
            "offset_y": round(self.Y_OFFSET_PER_TORSO_HEIGHT * torso_height) + micro_y,
            "opacity": 1.0,
        }

    def drape_saree(
        self,
        customer_pil: Image.Image,
        saree_pil: Image.Image,
        pose_data: dict,
        micro_scale: float = 1.0,
        micro_x: int = 0,
        micro_y: int = 0,
    ) -> tuple[Image.Image, dict]:
        """Resize, rotate, shoulder-anchor, and alpha-composite the saree."""
        metrics = pose_data["metrics"]
        landmarks = pose_data["landmarks"]
        calibration = self.automatic_calibration(metrics, micro_scale, micro_x, micro_y)

        target_width = max(1, int(
            metrics["shoulder_width_px"] * 2.2 * calibration["scale_mult"]
        ))
        source_width, source_height = saree_pil.size
        target_height = max(1, int(target_width * source_height / source_width))

        resized_saree = saree_pil.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        rotated_saree = resized_saree.rotate(
            -metrics["shoulder_tilt_deg"],
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )

        # MediaPipe's left means the customer's anatomical left shoulder.
        shoulder = landmarks["left_shoulder"]
        paste_x = (
            shoulder[0]
            - int(target_width * self.SAREE_SHOULDER_ANCHOR_X)
            + calibration["offset_x"]
        )
        paste_y = (
            shoulder[1]
            - int(target_height * self.SAREE_SHOULDER_ANCHOR_Y)
            + calibration["offset_y"]
        )

        output = customer_pil.convert("RGBA")
        output.paste(rotated_saree, (paste_x, paste_y), mask=rotated_saree)

        calibration.update({
            "target_width_px": target_width,
            "target_height_px": target_height,
            "paste_x": paste_x,
            "paste_y": paste_y,
        })
        return output.convert("RGB"), calibration


def get_customer_image(asset_dir: str, input_source: str):
    """Read the selected customer source, returning None until one is supplied."""
    sample_path = os.path.join(asset_dir, "sample_customer.jpg")
    if input_source == "Sample Model Photo":
        if not os.path.exists(sample_path):
            st.error(f"Sample image is missing: {sample_path}")
            return None
        return Image.open(sample_path)

    if input_source == "Upload Photo":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a clear front-facing upper-body photo", type=["jpg", "jpeg", "png"]
        )
        return Image.open(uploaded_file) if uploaded_file else None

    camera_file = st.sidebar.camera_input("Capture pose photo")
    return Image.open(camera_file) if camera_file else None


def run_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Run: pip install -r requirements.txt")
        return
    if not MEDIAPIPE_AVAILABLE:
        st.error("MediaPipe is not installed. Run: pip install -r requirements.txt")
        return

    st.set_page_config(page_title="Virtual Mirror — Pose Diagnostics", layout="wide")
    st.title("Virtual Mirror Saree Draping — Pose Diagnostics")
    st.caption("Automatic body-relative calibration with optional minor template corrections.")

    engine = VirtualMirrorEngine()
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")

    st.sidebar.header("1. Garment Catalog")
    saree_choice = st.sidebar.selectbox(
        "Select Saree Design",
        ["Kanchipuram Crimson & Gold", "Royal Peacock Blue & Gold"],
    )
    saree_filename = "saree_crimson.png" if "Crimson" in saree_choice else "saree_blue.png"
    saree_path = os.path.join(asset_dir, saree_filename)
    if not os.path.exists(saree_path):
        st.error(f"Saree asset is missing: {saree_path}")
        return
    saree_img = Image.open(saree_path).convert("RGBA")
    st.sidebar.image(saree_img, caption=saree_choice, use_container_width=True)

    st.sidebar.header("2. Customer Input")
    input_source = st.sidebar.radio(
        "Input Type", ["Sample Model Photo", "Upload Photo", "Webcam Capture"]
    )
    customer_img = get_customer_image(asset_dir, input_source)
    if customer_img is None:
        st.info("Select, upload, or capture a customer image to begin.")
        return

    st.sidebar.header("3. Optional Micro Calibration")
    st.sidebar.caption("Leave at zero for automatic calibration. Use only for a garment-template crop difference.")
    micro_scale = st.sidebar.slider("Micro Scale Trim", 0.85, 1.15, 1.00, 0.01)
    micro_x = st.sidebar.slider("Micro Horizontal Tweak (X)", -50, 50, 0, 1)
    micro_y = st.sidebar.slider("Micro Vertical Tweak (Y)", -50, 50, 0, 1)
    show_landmarks = st.sidebar.checkbox("Overlay Pose Landmarks Skeleton", value=False)

    customer_np = np.array(customer_img.convert("RGB"))
    pose_data, error = engine.extract_landmarks(customer_np)
    input_col, output_col = st.columns(2)

    with input_col:
        st.subheader("Input Customer Photo")
        if pose_data and show_landmarks:
            debug_img = customer_np.copy()
            engine.mp_drawing.draw_landmarks(
                debug_img,
                pose_data["raw_results"].pose_landmarks,
                engine.mp_pose.POSE_CONNECTIONS,
            )
            st.image(debug_img, use_container_width=True)
        else:
            st.image(customer_img, use_container_width=True)

    with output_col:
        st.subheader("Automatically Calibrated Output")
        if error:
            st.warning(error)
            st.image(customer_img, use_container_width=True)
            return

        draped_result, calibration = engine.drape_saree(
            customer_pil=customer_img,
            saree_pil=saree_img,
            pose_data=pose_data,
            micro_scale=micro_scale,
            micro_x=micro_x,
            micro_y=micro_y,
        )
        st.image(draped_result, use_container_width=True)

        metrics = pose_data["metrics"]
        st.markdown(f"""
        #### Pose and Frame Diagnostics
        - **Frame:** `{metrics['canvas_width']} × {metrics['canvas_height']} px`
        - **Shoulder span:** `{metrics['shoulder_width_px']} px`
        - **Hip span:** `{metrics['hip_width_px']} px`
        - **Torso span:** `{metrics['torso_height_px']} px`
        - **Shoulder tilt:** `{metrics['shoulder_tilt_deg']}°`
        - **Hip / shoulder ratio:** `{metrics['hip_to_shoulder_ratio']}`
        - **Torso / shoulder ratio:** `{metrics['torso_to_shoulder_ratio']}`

        #### Automatic Drape Values
        - **Target garment size:** `{calibration['target_width_px']} × {calibration['target_height_px']} px`
        - **Computed placement point:** `(X: {calibration['paste_x']}, Y: {calibration['paste_y']})`
        - **Scale multiplier:** `{calibration['scale_mult']:.2f}`
        - **Asset-alignment offset:** `(X: {calibration['offset_x']} px, Y: {calibration['offset_y']} px)`
        - **Garment opacity:** `1.00`
        """)


if __name__ == "__main__":
    run_app()
