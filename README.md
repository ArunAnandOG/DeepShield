# 🛡️ DeepShield – AI-Powered Deepfake Video Detection System

**DeepShield** is an AI-powered deepfake video detection and authenticity verification system developed using Deep Learning and Computer Vision. The project aims to identify manipulated videos by analyzing facial regions and extracting visual features to distinguish between authentic and AI-generated or manipulated content.

The system allows users to upload video files and receive an automated prediction indicating whether the video is **Real or Fake**.

## 🚀 Key Features

- **Video Upload:** Upload video files for deepfake analysis.
- **Face Detection:** Detect and extract facial regions from video frames.
- **Frame Processing:** Extract and preprocess video frames for analysis.
- **Deep Learning Classification:** Analyze extracted features using a trained deep learning model.
- **Real/Fake Prediction:** Classify videos as authentic or manipulated.
- **Result Display:** Present the detection results through a simple web interface.
- **Database Integration:** Store relevant detection records using SQLite.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development and inference |
| OpenCV | Video processing and face detection |
| MobileNetV2 / CNN | Deep learning-based feature extraction and classification |
| Flask | Backend web application framework |
| SQLite | Database management |
| HTML5 & CSS3 | Frontend development |
| JavaScript | Frontend interactions |
| Bootstrap 5 | Responsive user interface |

## ⚙️ How It Works

1. The user uploads a video through the web application.
2. The system extracts frames from the uploaded video.
3. Face detection is performed to identify facial regions.
4. Detected faces are preprocessed and resized to the required input dimensions.
5. The trained deep learning model analyzes the extracted facial information.
6. The system generates a prediction.
7. The final result is displayed as **Real** or **Fake**.

## 🏗️ Project Architecture

The system consists of the following major components:

- **Frontend:** Provides the interface for uploading videos and viewing results.
- **Backend:** Handles video uploads, processing requests, and communication between modules.
- **Face Detection Module:** Identifies facial regions within video frames.
- **Deep Learning Module:** Performs feature extraction and classification.
- **Database Module:** Manages stored detection information.

## 💻 Installation and Setup

### Prerequisites

- Python 3.x
- pip
- Visual Studio Code or PyCharm

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/DeepShield.git
```

### 2. Navigate to the Project Directory

```bash
cd DeepShield
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

## 📂 Project Structure

```text
DeepShield/
│
├── app.py
├── detect_deepfake.py
├── model/
│   └── model.h5
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── requirements.txt
└── README.md
```

*Note: The directory structure above is illustrative. Adjust it to match the actual repository.*

## 🎯 Project Objectives

- Develop an automated deepfake video detection system.
- Apply deep learning techniques to identify manipulated facial content.
- Explore computer vision techniques for video frame processing and face detection.
- Provide a simple and accessible platform for video authenticity analysis.
- Contribute to awareness of digital media manipulation and misinformation.

## 🔮 Future Enhancements

- Improve detection accuracy using larger and more diverse datasets.
- Integrate advanced spatial and temporal feature extraction techniques.
- Support additional video formats and higher-resolution videos.
- Optimize model inference for faster processing.
- Explore transformer-based architectures for improved deepfake detection.

## 🎓 Academic Project

Developed as part of an **MSc Computer Science** academic project.

## 📜 License

This project is intended for educational and research purposes.
