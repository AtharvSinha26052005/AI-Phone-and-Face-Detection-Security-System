# Import necessary libraries from Flask
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response

# Secure file upload handling
from werkzeug.utils import secure_filename

# Import YOLOv8 from Ultralytics
from ultralytics import YOLO

# Import essential modules
import os, cv2, json

# Initialize Flask application
app = Flask(__name__)

# Load the trained YOLO model
model = YOLO("best.pt")

# Define folder for uploading and storing images
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Set upload folder path in app config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for landing/home page
@app.route('/')
def index():
    # Render the landing page
    return render_template('landin.html')

# Route to upload and process image
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    # If POST method (form submitted)
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files.get('image')
        # Check if file exists and has a name
        if file and file.filename:
            # Secure the filename
            filename = secure_filename(file.filename)
            # Create path to save the uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            # Save file to upload folder
            file.save(filepath)

            # Run YOLO object detection on the image
            results = model.predict(source=filepath, conf=0.5)
            result = results[0]

            # Draw bounding boxes and labels
            annotated = result.plot()
            # Create result image name and path
            result_img_name = 'result_' + filename
            result_img_path = os.path.join(UPLOAD_FOLDER, result_img_name)
            # Save annotated image
            cv2.imwrite(result_img_path, annotated)

            # Initialize list to store JSON data
            json_data = []
            # Define path for text file output
            txt_path = os.path.join(DETECTION_FOLDER, filename.rsplit('.', 1)[0] + '.txt')
            # Write detection results to .txt file
            with open(txt_path, 'w') as f:
                for box in result.boxes:
                    # Get class, confidence, and coordinates
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[cls]
                    # Append detection info to JSON list
                    json_data.append({
                        'label': label,
                        'confidence': round(conf, 2),
                        'bbox': [x1, y1, x2, y2]
                    })
                    # Write to text file
                    f.write(f"{label} {conf:.2f} {x1} {y1} {x2} {y2}\n")

            # Write detection results to .json file
            json_path = os.path.join(DETECTION_FOLDER, filename.rsplit('.', 1)[0] + '.json')
            with open(json_path, 'w') as jf:
                json.dump(json_data, jf, indent=2)

            # Render result page and pass image & file info
            return render_template('result.html',
                                   result_image='uploads/' + result_img_name,
                                   txt_file=os.path.basename(txt_path),
                                   json_file=os.path.basename(json_path))
    # If GET method, show upload form
    return render_template('upload.html')

# Route to download result files (.txt, .json)
@app.route('/download/<filename>')
def download_file(filename):
    # Send requested file from detection folder
    return send_from_directory(DETECTION_FOLDER, filename, as_attachment=True)

# Flag to pause webcam detection
paused = False

# Route to toggle webcam pause/resume
@app.route('/toggle_pause')
def toggle_pause():
    global paused
    # Switch pause state
    paused = not paused
    return ('', 204)  # Return no content

# Generator function to yield webcam frames
def generate_frames():
    # Open webcam
    cap = cv2.VideoCapture(0)
    global paused
    while True:
        # Read a frame
        success, frame = cap.read()
        if not success:
            break
        # If not paused, run YOLO detection
        if not paused:
            results = model.predict(source=frame, conf=0.5)
            frame = results[0].plot()
        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        # Yield frame as byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Route to render webcam page
@app.route('/live')
def live():
    return render_template('live.html')

# Route to serve index.html directly if needed
@app.route('/index.html')
def index_page():
    return render_template('index.html')

# Route for video feed using multipart streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app in debug mode when executed directly
if __name__ == '__main__':
    app.run(debug=True)
