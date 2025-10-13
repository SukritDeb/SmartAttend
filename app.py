import base64
import io
import os
import shutil
import csv
import time
import datetime
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template, request, jsonify
from PIL import Image


app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
required_dirs = [
    "Attendance",
    "StudentDetails",
    "TrainingImage",
    "TrainingImageLabel",
]
for d in required_dirs:
    path = os.path.join(base_dir, d)
    if not os.path.exists(path):
        os.makedirs(path)


haar_cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
student_details_path = os.path.join(base_dir, "StudentDetails", "studentdetails.csv")
training_image_path = os.path.join(base_dir, "TrainingImage")
training_label_path = os.path.join(base_dir, "TrainingImageLabel", "Trainner.yml")
attendance_path = os.path.join(base_dir, "Attendance")



def decode_image(image_data):
    """Decodes a base64 image string to an OpenCV-readable format."""
    if "," in image_data:
        image_data = image_data.split(",")[1]
    decoded_bytes = base64.b64decode(image_data)
    pil_image = Image.open(io.BytesIO(decoded_bytes))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



def train_model():
    """Trains the face recognizer."""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = get_images_and_labels(training_image_path)
        if not faces:
            return False, "No faces to train."
        recognizer.train(faces, np.array(ids))
        recognizer.save(training_label_path)
        return True, "Model trained successfully!"
    except Exception as e:
        return False, f"Error during training: {str(e)}"

def get_images_and_labels(path):
    """Gets all face images and their IDs."""
    faces, ids = [], []
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            for image_file in os.listdir(subdir_path):
                if image_file.endswith((".jpg", ".png")):
                    image_path = os.path.join(subdir_path, image_file)
                    pil_image = Image.open(image_path).convert("L")
                    image_np = np.array(pil_image, "uint8")
                    try:
                        student_id = int(os.path.basename(image_file).split("_")[1])
                        faces.append(image_np)
                        ids.append(student_id)
                    except (IndexError, ValueError):
                        continue
    return faces, ids


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/view')
def view_page():
    """Renders the page to view attendance, requires subject selection."""
    subject = request.args.get('subject')
    if not subject:
        return render_template('view.html', subject_form=True)

    # --- Analytics logic from showattedance.py ---
    subject_folder = os.path.join(attendance_path, subject)
    if not os.path.exists(subject_folder):
        return render_template('view.html', subject_form=True, message=f"No records found for subject: {subject}")

    all_files = [f for f in os.listdir(subject_folder) if f.endswith('.csv')]
    if not all_files:
        return render_template('view.html', subject_form=True, message=f"No attendance sheets found for {subject}.")
        
    df_students = pd.read_csv(student_details_path)
    merged_df = df_students[['Enrollment', 'Name']].copy()

    # Read and merge each attendance file
    for file in all_files:
        date_str = file.split('_')[1]
        df_daily = pd.read_csv(os.path.join(subject_folder, file))
        # Use the date column from the file, not the filename
        date_col_name = df_daily.columns[2]
        merged_df = pd.merge(merged_df, df_daily[['Enrollment', date_col_name]], on='Enrollment', how='left')
    
    merged_df.fillna(0, inplace=True) # Fill missing attendance with 0 (Absent)
    
    # Calculate totals
    attendance_cols = merged_df.columns[2:]
    merged_df['Total_Present'] = merged_df[attendance_cols].sum(axis=1)
    
    total_sessions = len(attendance_cols)
    if total_sessions > 0:
        merged_df['Attendance_%'] = ((merged_df['Total_Present'] / total_sessions) * 100).round().astype(int)
    else:
        merged_df['Attendance_%'] = 0

    headers = list(merged_df.columns)
    records = merged_df.values.tolist()
    
    return render_template('view.html', subject_form=True, subject=subject, headers=headers, records=records)

# --- API Routes for Functionality ---
@app.route('/register_student', methods=['POST'])
def register_student():
    """Handles student registration: saves details, face images, and retrains the model."""
    try:
        name = request.form.get('name')
        student_id = request.form.get('student_id')
        image_data = request.form.get('image_data')

        if not all([name, student_id, image_data]):
            return jsonify({"success": False, "message": "Missing form data."}), 400
        
        # Save Student Details
        with open(student_details_path, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([int(student_id), name])

        # Decode and Save Face Images
        img = decode_image(image_data)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(haar_cascade_path)
        faces = detector.detectMultiScale(gray_img, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"success": False, "message": "No face detected in the image."})

        student_dir = os.path.join(training_image_path, f"{student_id}_{name}")
        os.makedirs(student_dir, exist_ok=True)
        
        face_img = gray_img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
        for sample_num in range(1, 51):
            file_name = f"{name}_{student_id}_{sample_num}.jpg"
            cv2.imwrite(os.path.join(student_dir, file_name), face_img)

        # Retrain the model
        success, message = train_model()
        if not success:
             return jsonify({"success": False, "message": message})

        return jsonify({"success": True, "message": f"Student {name} registered! Model retrained."})

    except Exception as e:
        return jsonify({"success": False, "message": f"Server Error: {e}"}), 500

@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    """Recognizes faces in a single video frame."""
    if not os.path.exists(training_label_path):
        return jsonify({"faces": [], "message": "Model not found. Please register a student first."})

    try:
        image_data = request.json.get('image_data')
        img = decode_image(image_data)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(training_label_path)
        detector = cv2.CascadeClassifier(haar_cascade_path)
        df_students = pd.read_csv(student_details_path)
        
        faces = detector.detectMultiScale(gray_img, 1.2, 5)
        recognized_faces = []

        for (x, y, w, h) in faces:
            student_id, conf = recognizer.predict(gray_img[y:y+h, x:x+w])
            
            name = "Unknown"
            if conf < 70: # Confidence threshold
                try:
                    name = df_students.loc[df_students['Enrollment'] == student_id]['Name'].values[0]
                except IndexError:
                    name = "Known Face (ID error)"

            
           
            recognized_faces.append({
                "id": int(student_id) if name != "Unknown" else -1,
                "name": name,
                "box": [int(x), int(y), int(w), int(h)]
            })
            
            
        return jsonify({"faces": recognized_faces})

    except Exception as e:
        return jsonify({"faces": [], "error": str(e)})

@app.route('/save_attendance_session', methods=['POST'])
def save_attendance_session():
    """Saves the final attendance sheet for a subject session."""
    data = request.get_json()
    subject = data.get('subject')
    present_ids = data.get('present_ids')

    if not subject or present_ids is None:
        return jsonify({"success": False, "message": "Missing subject or student data."})

    try:
        df_all_students = pd.read_csv(student_details_path)
        
        # Generate a unique timestamp for the session
        session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        attendance_column_name = session_timestamp
        
        df_all_students[attendance_column_name] = 0 # Default to absent
        
        # Mark present students
        df_all_students.loc[df_all_students['Enrollment'].isin(present_ids), attendance_column_name] = 1

        # Save file
        subject_folder = os.path.join(attendance_path, subject)
        os.makedirs(subject_folder, exist_ok=True)
        
        file_name = f"{subject}_{session_timestamp}.csv"
        df_all_students.to_csv(os.path.join(subject_folder, file_name), index=False)

        return jsonify({"success": True, "message": f"Attendance for {subject} saved successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server Error: {str(e)}"})

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(student_details_path) or os.path.getsize(student_details_path) == 0:
        with open(student_details_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Enrollment', 'Name'])
            
    app.run(debug=True, port=5001)