from flask import Flask, render_template, request, redirect, send_from_directory
from saveUpload_db import save_upload
from saveResult_db import save_result
from detect_deepfake import detect_deepfake
from init_db import get_filepath_from_db, get_result_from_db
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

progress_status = {}

# Home
@app.route('/')
def home():
    return render_template('index.html')

# Upload
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        upload_id = save_upload(file.filename, filepath)

        return redirect(f'/process/{upload_id}')

    return "No file uploaded"

# Processing Page (HTML)
@app.route('/process/<int:upload_id>')
def process(upload_id):
    return render_template('processing.html', upload_id=upload_id)

# Start ML Processing
@app.route('/start_processing/<int:upload_id>')
def start_processing(upload_id):
    progress_status[upload_id] = 0

    filepath = get_filepath_from_db(upload_id)

    progress_status[upload_id] = 20
    time.sleep(1)

    progress_status[upload_id] = 50
    prediction, confidence = detect_deepfake(filepath)

    progress_status[upload_id] = 80
    save_result(upload_id, prediction, confidence)

    progress_status[upload_id] = 100

    return "Done"

# Progress API
@app.route('/progress/<int:upload_id>')
def progress(upload_id):
    return {"progress": progress_status.get(upload_id, 0)}

# Result Page
@app.route('/result/<int:upload_id>')
def result(upload_id):
    prediction, confidence = get_result_from_db(upload_id)
    filepath = get_filepath_from_db(upload_id)
    filename = os.path.basename(filepath)

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=(confidence * 100),
        filename=filename
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# RUN (ALWAYS LAST)
if __name__ == "__main__":
    app.run(debug=True)