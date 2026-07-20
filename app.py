import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os



st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Road Damage Detection using YOLO")



import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "YOLO11n": os.path.join(BASE_DIR, "YOLO11n_best.pt"),
    "YOLO11s": os.path.join(BASE_DIR, "YOLO11s_best.pt"),
    "YOLO11m": os.path.join(BASE_DIR, "YOLO11m_best.pt"),
    "YOLO11s Tuned ": os.path.join(BASE_DIR, "YOLO11s_Tuned_best.pt"),
}
model_name = st.sidebar.selectbox(
    "Choose Model",
    list(MODELS.keys())
)

st.sidebar.title("⚙ Detection Settings")

model_name = st.sidebar.selectbox(
    " Select YOLO Model",
    list(MODELS.keys())
)

confidence = st.sidebar.slider(
    "🎯 Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

mode = st.sidebar.radio(
    " Input Type",
    ["Image", "Video"]
)

st.sidebar.markdown("---")



@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODELS[model_name])



if mode == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        img = np.array(image)

        results = model.predict(
            img,
            conf=confidence
        )

        annotated = results[0].plot()

        with col2:
            st.subheader("Detection")
            st.image(
                annotated,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Detected Objects")

        boxes = results[0].boxes

        if len(boxes) == 0:

            st.warning("No Road Damage Detected")

        else:

            from collections import Counter

            counter = Counter()

            data = []

            total = 0

            for box in boxes:

                cls = int(box.cls)

                conf = float(box.conf)

                class_name = model.names[cls]

                counter[class_name] += 1

                data.append(
                    {
                        "Damage Type": class_name,
                        "Confidence": round(conf, 2)
                    }
                )

                total += 1

            st.markdown("---")

            st.subheader(" Detection Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(" Potholes", counter["Pothole"])

            with col2:
                st.metric(" Cracks", counter["Crack"])

            with col3:
                st.metric(" Manholes", counter["Manhole"])

            with col4:
                st.metric(" Total", total)

            st.markdown("---")

            st.subheader(" Detection Table")

            st.dataframe(
                data,
                use_container_width=True
            )
elif mode == "Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs("outputs", exist_ok=True)

        output_path = "outputs/result.mp4"

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_placeholder = st.empty()

        progress = st.progress(0)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                frame,
                conf=confidence,
                verbose=False
            )

            annotated = results[0].plot()

            writer.write(annotated)

            frame_placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

            frame_number += 1

            if total_frames > 0:
                progress.progress(
                    frame_number / total_frames
                )

        cap.release()

        writer.release()

        st.success("Video Finished Successfully ")

        st.video(output_path)

        with open(output_path, "rb") as file:

            st.download_button(
                label="⬇ Download Result",
                data=file,
                file_name="RoadDamage_Result.mp4",
                mime="video/mp4"
            )