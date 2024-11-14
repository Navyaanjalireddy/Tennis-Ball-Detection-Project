import streamlit as st
import cv2
import torch
from pathlib import Path
import tempfile
import sys
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
from tqdm import tqdm
import numpy as np

# Function to plot one bounding box on the image
def plot_one_box(xyxy, img, color=(255, 0, 0), label=None, line_thickness=2):
    """Draws one bounding box on an image."""
    tl = max(int(line_thickness), 1)  # line thickness
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

st.title("YOLOv5 Video Detection Application")

# File uploader for video input
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    # Create a temporary file to store the uploaded video
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path.write(uploaded_video.read())
    temp_video_path.close()

    # Process button
    if st.button("Process"):
        # Load video and initialize parameters
        cap = cv2.VideoCapture(temp_video_path.name)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.write("Processing video...")

        # Progress bar
        progress_bar = st.progress(0)

        # Load YOLOv5 model directly using torch.hub.load
        model_weights_path = 'runs/train/exp/weights/best.pt'

        # Make sure to run the model loading in your environment with 'yolov5' available
        model = torch.hub.load('yolov5', 'custom', path=model_weights_path, source='local')

        # Set model to evaluation mode
        model.eval()

        # Process each frame
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and normalize frame to match model input
            resized_frame = cv2.resize(frame, (640, 640))
            results = model(resized_frame)

            # Extract bounding boxes and labels
            for *xyxy, conf, cls in results.xyxy[0].tolist():
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)

            # Write frame to output
            out.write(frame)

            # Update progress
            progress_percentage = int((i + 1) / total_frames * 100)
            progress_bar.progress(progress_percentage)

        # Release resources
        cap.release()
        out.release()

        st.success("Detection complete!")

        # Display processed video
        st.video(output_path)