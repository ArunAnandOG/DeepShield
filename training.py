import os
import cv2
import numpy as np
import tensorflow as tf

# -------------------------------
# CONFIG
# -------------------------------
VIDEO_DATASET = "videos"
FRAME_DATASET = "frames_dataset"
IMG_SIZE = 224
FRAMES_PER_VIDEO = 50
BATCH_SIZE = 16
EPOCHS = 10

# -------------------------------
# LOAD FACE DETECTOR
# -------------------------------
face_net = cv2.dnn.readNetFromCaffe(
    "model/deploy.prototxt",
    "model/res10_300x300_ssd_iter_140000.caffemodel"
)

# -------------------------------
# STEP 1: Extract FACE frames
# -------------------------------
def extract_frames():
    classes = ["real", "fake"]
    

    for cls in classes:
        video_folder = os.path.join(VIDEO_DATASET, cls)
        output_folder = os.path.join(FRAME_DATASET, cls)
        os.makedirs(output_folder, exist_ok=True)

        for video_file in os.listdir(video_folder):
            print("Video Selected")
            if not video_file.endswith(".mp4"):
                continue

            cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // FRAMES_PER_VIDEO)

            count, saved = 0, 0

            while cap.isOpened() and saved < FRAMES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % step == 0:
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

                    if best_face is not None:
                        face = cv2.resize(best_face, (IMG_SIZE, IMG_SIZE))
                        name = f"{video_file}_{saved}.jpg"
                        cv2.imwrite(os.path.join(output_folder, name), face)
                        saved += 1

                count += 1

            cap.release()

    print("✅ Face frames extracted")

# -------------------------------
# STEP 2: Load dataset
# -------------------------------
def get_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        FRAME_DATASET,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    print(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        FRAME_DATASET,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(2000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds

# -------------------------------
# STEP 3: Build model
# -------------------------------
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.base_model = base_model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

# -------------------------------
# STEP 4: Train
# -------------------------------
def train_model():
    # Avoid duplicate extraction
    if not os.path.exists(FRAME_DATASET):
        extract_frames()

    train_ds, val_ds = get_datasets()
    model, base_model = build_model()

    # Early stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("🚀 Initial Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[callback]
    )

    # -------------------------------
    # Fine-tuning
    # -------------------------------
    print("🔁 Fine-tuning...")

    base_model = model.base_model  # MobileNetV2
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[callback]
    )

    os.makedirs("model", exist_ok=True)
    model.save("model/model_v3.h5")

    print("✅ Training complete & model saved!")

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    train_model()