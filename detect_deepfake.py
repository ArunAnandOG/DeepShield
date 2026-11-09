import cv2
import numpy as np
import tensorflow as tf

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("model/model_v3.h5")

# -------------------------------
# LOAD FACE DETECTOR
# -------------------------------
face_net = cv2.dnn.readNetFromCaffe(
    "model/deploy.prototxt",
    "model/res10_300x300_ssd_iter_140000.caffemodel"
)

IMG_SIZE = 224

# -------------------------------
# EXTRACT FRAMES
# -------------------------------
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    frames = []
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames

# -------------------------------
# DETECT DEEPFAKE
# -------------------------------
def detect_deepfake(video_path):
    frames = extract_frames(video_path)

    scores = []

    for frame in frames:
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)
        detections = face_net.forward()

        best_face = None
        best_conf = 0

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]

            if conf > 0.5 and conf > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (l, t, r, b) = box.astype("int")

                pad = 20
                l = max(0, l - pad)
                t = max(0, t - pad)
                r = min(w, r + pad)
                b = min(h, b + pad)

                face = frame[t:b, l:r]

                if face.size > 0:
                    best_face = face
                    best_conf = conf

        if best_face is None:
            continue

        face = cv2.resize(best_face, (IMG_SIZE, IMG_SIZE))
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0][0]
        scores.append(pred)

    # -------------------------------
    # DECISION LOGIC (FRAME VOTING)
    # -------------------------------
    scores = np.array(scores)

    if len(scores) == 0:
        return "NO_FACE", 0.0

    avg_score = np.mean(scores)
    real_ratio = np.sum(scores > 0.5) / len(scores)
    max_score = np.max(scores)

    # RULES
    if max_score < 0.3:
        label = "FAKE"
    elif real_ratio > 0.7 and avg_score > 0.6:
        label = "REAL"
    else:
        label = "FAKE"

    confidence = float(avg_score if label == "REAL" else 1 - avg_score)

    return label, round(confidence, 4)