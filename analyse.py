import os
from detect_deepfake import detect_deepfake
from sklearn.metrics import confusion_matrix, classification_report

VIDEO_DATASET = "videos"

def evaluate():
    total = 0
    correct = 0

    y_true = []
    y_pred = []

    for label in ["real", "fake"]:
        folder = os.path.join(VIDEO_DATASET, label)

        for file in os.listdir(folder):
            if not file.endswith(".mp4"):
                continue

            video_path = os.path.join(folder, file)

            print(f"\nProcessing: {file} ({label})")

            pred_label, conf = detect_deepfake(video_path)

            print(f"Predicted: {pred_label} | Confidence: {conf}")

            total += 1

            # Normalize labels
            true_label = label.upper()
            pred_label = pred_label.upper()

            y_true.append(true_label)
            y_pred.append(pred_label)

            if pred_label == true_label:
                correct += 1

    accuracy = correct / total if total > 0 else 0

    print("\n==========================")
    print(f"Total Videos: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("==========================")
    
    # CONFUSION MATRIX
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return y_true, y_pred


if __name__ == "__main__":
    evaluate()