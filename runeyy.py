import os
import csv
from detect_deepfake import detect_deepfake

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_FOLDER = "videos/real"   # folder with videos
OUTPUT_FILE = "real.csv"

# -----------------------------
# PROCESS VIDEOS
# -----------------------------
def process_videos():
    results = []

    for file in os.listdir(VIDEO_FOLDER):
        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEO_FOLDER, file)
        print(f"Processing: {file}")

        try:
            label, conf, avg, var, diff = detect_deepfake(video_path)

            results.append({
                "video": file,
                "label": label,
                "confidence": conf,
                "avg": avg,
                "var": var,
                "diff": diff
            })

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return results

# -----------------------------
# SAVE RESULTS
# -----------------------------
def save_results(results):
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "confidence", "avg", "var", "diff"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Results saved to {OUTPUT_FILE}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    results = process_videos()
    save_results(results)