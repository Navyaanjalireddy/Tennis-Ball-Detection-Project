# 🎾Tennis Ball Tracking and Region Detection

A computer vision-based tennis analysis project that uses **YOLOv5** to detect and track tennis players and the tennis ball from video footage. The system processes tennis match videos frame by frame and identifies objects using a custom-trained YOLOv5 model.

The project also includes a **Streamlit web application** that allows users to upload a tennis video, process it using the trained detection model, visualize the detection results, and download the processed video.

---

## 📌 Project Overview

Analyzing tennis videos manually can be time-consuming, especially when identifying the position of players and tracking the tennis ball across frames.

This project aims to automate this process using **Deep Learning and Computer Vision** techniques. A custom dataset was prepared and annotated for tennis-related objects, converted into the YOLO format, and used to train a YOLOv5 object detection model.

The trained model can identify:

* 🎾 Tennis Ball
* 🧍 Player 1
* 🧍 Player 2
* 🎾 Tennis Court / Court Regions

The detected objects are displayed using bounding boxes and confidence scores on the processed video.

---

## 🎯 Objectives

The main objectives of this project are:

1. Detect tennis players from video footage.
2. Detect the tennis ball in each video frame.
3. Identify tennis court boundaries and regions.
4. Track the movement of detected objects across video frames.
5. Train a custom YOLOv5 object detection model.
6. Build a user-friendly interface for video processing.
7. Generate and download the processed detection video.

---

## 🧠 Technologies Used

| Technology    | Purpose                    |
| ------------- | -------------------------- |
| Python        | Main programming language  |
| YOLOv5        | Object detection           |
| PyTorch       | Deep learning framework    |
| OpenCV        | Video and image processing |
| NumPy         | Numerical operations       |
| Streamlit     | Web application            |
| Google Colab  | Model training             |
| VGG Annotator | Dataset annotation         |

---

## 🔄 Project Workflow

```text
Input Tennis Video
        ↓
Extract Video Frames
        ↓
Annotate Tennis Objects
        ↓
Convert Annotations to YOLO Format
        ↓
Prepare Custom Dataset
        ↓
Train YOLOv5 Model
        ↓
Validate Trained Model
        ↓
Detect Players & Tennis Ball
        ↓
Region / Court Detection
        ↓
Generate Processed Video
        ↓
Streamlit Web Application
```

---

## 📂 Project Structure

```text
Tennis-Ball-Detection-Project/
│
├── classify/
├── data/
├── models/
├── runs/
│   └── train/
│       └── exp/
│           └── weights/
│               └── best.pt
│
├── segment/
├── utils/
│
├── app.py
├── detect.py
├── train.py
├── val.py
├── export.py
├── benchmarks.py
│
├── requirements.txt
├── yolov5s.pt
├── tutorial.ipynb
├── pyproject.toml
└── README.md
```

---

## 🏷️ Dataset Preparation

The tennis video was first converted into individual frames.

The frames were then manually annotated using **VGG Annotator (VIA)**.

Annotations were prepared for the relevant tennis objects, including players, tennis ball, and court-related regions.

The annotations were then converted into the format required by YOLOv5.

The resulting dataset was organized into training and validation data before model training.

---

## 🤖 YOLOv5 Model Training

A custom YOLOv5 model was trained using the prepared tennis dataset.

The model was trained to recognize tennis-related objects from individual video frames.

The trained model weights are stored in:

```text
runs/train/exp/weights/best.pt
```

The repository also contains the YOLOv5 base model:

```text
yolov5s.pt
```

---

## 🎾 Object Detection

During inference, each frame of the uploaded video is passed through the trained YOLOv5 model.

The model generates:

* Bounding box coordinates
* Object class
* Confidence score

Non-Maximum Suppression is applied to remove overlapping detections.

Detected objects are then displayed on the video using bounding boxes and labels.

Example detection format:

```text
player1 0.91
player2 0.88
ball    0.76
```

---

## 📍 Region Detection

The project also focuses on identifying the position of players with respect to the tennis court.

Player annotations contain region information that can be used to distinguish between the two sides of the court.

This allows the system to determine which region of the court a detected player belongs to and provides a foundation for further tennis analytics.

---

## 🖥️ Streamlit Application

The project includes a Streamlit-based interface for running object detection on tennis videos.

The application allows the user to:

1. Upload a video.
2. Select **Process**.
3. Process the video frame by frame.
4. View the detection progress.
5. Watch the processed video.
6. Download the processed video.

Supported input formats include:

```text
.mp4
.mov
.avi
.mkv
```

The Streamlit application uses the trained model located at:

```text
runs/train/exp/weights/best.pt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Navyaanjalireddy/Tennis-Ball-Detection-Project.git
```

### 2. Navigate to the project

```bash
cd Tennis-Ball-Detection-Project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a tennis video and click:

```text
Process
```

The application will process the video and display the detection results.

---

## 🔍 Running YOLOv5 Detection Directly

The YOLOv5 detection script can also be used directly:

```bash
python detect.py
```

Modify the required input source and model path according to your setup.

---

## 📊 Output

The final output is a processed tennis video containing:

* Player bounding boxes
* Tennis ball detection
* Object labels
* Confidence scores
* Frame-by-frame detection results

The Streamlit application also provides an option to download the processed video.

---

## 🚀 Future Enhancements

The project can be extended with additional tennis analytics such as:

* 🎾 Advanced tennis ball trajectory tracking
* 📈 Ball speed estimation
* 📍 Player movement tracking
* 🏃 Player distance covered
* 🗺️ Court heatmap generation
* 🎯 Shot and bounce detection
* 📊 Match statistics
* 🔄 Real-time camera-based detection
* 🧠 Improved tracking using DeepSORT/ByteTrack
* 📱 Deployment as a cloud-based application

---

## 💡 Applications

This system can be used for:

* Tennis match analysis
* Player movement analysis
* Sports analytics
* Training and coaching
* Automated video analysis
* Computer vision research
* AI-based sports applications

---

## 👩‍💻 Author

**G Navyaanjali Reddy**

Computer Science & Engineering (Artificial Intelligence & Machine Learning)

---

## ⭐ Acknowledgements

This project uses the **YOLOv5** object detection framework and related computer vision libraries for training and inference.
