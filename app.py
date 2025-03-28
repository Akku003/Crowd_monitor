from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from flask_pymongo import PyMongo
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import math
from twilio.rest import Client
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
from threading import Lock
import json
import base64
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MongoDB Configuration
'''app.config["MONGO_URI"] = "mongodb://localhost:27017/crowd_monitoring"'''
app.config["MONGO_URI"] ="mongodb+srv://crowdmonitor:crowdmonitor@cluster0.raymtb1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Needed for session management
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB upload limit

# Email configuration
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_email_password'

# Twilio credentials
TWILIO_ACCOUNT_SID = 'your_twilio_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_token'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load YOLOv8 model
model = YOLO("yolo11m.pt")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'processed'), exist_ok=True)

# Global variables for real-time data
real_time_data = {
    'timestamps': [],
    'density_values': [],
    'count_values': [],
    'speed_values': [],
    'lock': Lock()
}

# In-memory user storage (for demo purposes - use a database in production)
users = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_email(to_email, subject, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

def send_whatsapp_alert(to_whatsapp_number, message):
    try:
        message = twilio_client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f'whatsapp:{to_whatsapp_number}'
        )
        logger.info(f"WhatsApp alert sent to {to_whatsapp_number}")
        return True
    except Exception as e:
        logger.error(f"Error sending WhatsApp alert: {e}")
        return False

def calculate_density(boxes, frame_area):
    if not boxes or frame_area == 0:
        return 0.0, 0
    
    # Calculate total area of all bounding boxes
    total_box_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_area = (x2 - x1) * (y2 - y1)
        total_box_area += box_area
    
    # Calculate raw density (percentage of frame covered)
    raw_density = total_box_area / frame_area
    
    # Apply more realistic normalization
    # Assuming each person occupies about 2-10% of frame area depending on distance
    # We'll use a dynamic normalization factor based on average person size
    avg_person_area = total_box_area / len(boxes) if boxes else 0
    normalization_factor = max(0.02, min(0.10, avg_person_area / frame_area))
    
    # Calculate normalized density (how many "standard persons" fit in the frame)
    normalized_density = raw_density / normalization_factor
    
    # Cap density at 1.0 (100%) and ensure it's not artificially high
    final_density = min(normalized_density, 1.0)
    
    # Apply smoothing for single person cases
    if len(boxes) == 1:
        final_density = min(final_density, 0.3)  # Max 30% for single person
    
    return final_density, len(boxes)

def get_density_level(density):
    if density < 0.1:    # 0-10% of normalized capacity
        return "Low"
    elif density < 0.3:  # 10-30%
        return "Medium"
    elif density < 0.6:  # 30-60%
        return "High"
    else:                # 60-100%
        return "Very High"

def calculate_speed(prev_frame_gray, frame_gray, prev_points):
    if prev_points is None or len(prev_points) == 0:
        return 0.0, None
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame_gray, frame_gray, prev_points, None, **lk_params)
    
    speeds = []
    valid_next_points = []

    for i, (new, old) in enumerate(zip(next_points, prev_points)):
        if status[i]:
            new_x, new_y = new.ravel()
            old_x, old_y = old.ravel()
            displacement = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
            speeds.append(displacement)
            valid_next_points.append([new_x, new_y])

    avg_speed = np.mean(speeds) if speeds else 0.0
    return avg_speed, np.array(valid_next_points, dtype=np.float32) if valid_next_points else None

def create_density_map(boxes, frame_shape):
    density_map = np.zeros(frame_shape[:2], dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= center_y < frame_shape[0] and 0 <= center_x < frame_shape[1]:
            density_map[center_y, center_x] += 1
    density_map = scipy.ndimage.gaussian_filter(density_map, sigma=15)
    return density_map

def process_single_frame(frame, crowd_limit):
    try:
        results = model(frame)
        boxes = [box for box in results[0].boxes if box.cls == 0]  # Class 0 is 'person'
        
        frame_area = frame.shape[0] * frame.shape[1]
        density, count = calculate_density(boxes, frame_area)
        
        crowd_limit_exceeded = count > crowd_limit
        
        # Create annotated frame
        annotated_frame = frame.copy()

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Create density map
        density_map = create_density_map(boxes, frame.shape)
        density_map_normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(density_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap, 0.3, 0)

        # Add text overlay
        y_offset = 30
        cv2.putText(annotated_frame, f'People Count: {count}', (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Density: {density:.2f} ({get_density_level(density)})', 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if crowd_limit_exceeded:
            cv2.putText(annotated_frame, 'Crowd Limit Exceeded!', (10, y_offset + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return {
            "success": True,
            "count": count,
            "density": density,
            "crowd_limit_exceeded": crowd_limit_exceeded,
            "annotated_frame": annotated_frame
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def process_video_with_density(video_path, crowd_limit):
    global real_time_data
    
    # Reset real-time data
    with real_time_data['lock']:
        real_time_data['timestamps'] = []
        real_time_data['density_values'] = []
        real_time_data['count_values'] = []
        real_time_data['speed_values'] = []
    
    cap = cv2.VideoCapture(video_path)
    crowd_limit_exceeded = False
    prev_frame_gray = None
    prev_points = None
    max_density = 0
    max_count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_area = frame_width * frame_height

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed', os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'mp4v' if 'avc1' doesn't work
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time
        
        # Convert to grayscale for optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO detection
        results = model(frame)
        boxes = [box for box in results[0].boxes if box.cls == 0]  # Class 0 is 'person'

        # Calculate density and count
        density, total_count = calculate_density(boxes, frame_area)
        
        # Update max values
        if density > max_density:
            max_density = density
        if total_count > max_count:
            max_count = total_count

        # Check crowd limit
        if total_count > crowd_limit:
            crowd_limit_exceeded = True

        # Speed calculation
        if prev_frame_gray is not None:
            if prev_points is None or len(prev_points) == 0:
                prev_points = cv2.goodFeaturesToTrack(
                    prev_frame_gray,
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7
                )
                if prev_points is not None:
                    prev_points = prev_points.reshape(-1, 1, 2)
                else:
                    prev_points = np.array([]).reshape(0, 1, 2)
            
            avg_speed, next_points = calculate_speed(prev_frame_gray, frame_gray, prev_points)
        else:
            avg_speed, next_points = 0.0, None

        # Update points for next frame
        if next_points is not None and len(next_points) > 0:
            prev_points = next_points.reshape(-1, 1, 2)
        else:
            prev_points = None

        # Update real-time data
        with real_time_data['lock']:
            real_time_data['timestamps'].append(current_time)
            real_time_data['density_values'].append(density)
            real_time_data['count_values'].append(total_count)
            real_time_data['speed_values'].append(avg_speed if avg_speed is not None else 0.0)

        # Density map and heatmap
        density_map = create_density_map(boxes, frame.shape)
        density_map_normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(density_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        # Annotate frame
        y_offset = 30
        cv2.putText(overlay, f'People Count: {total_count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, f'Density: {density:.2f} ({get_density_level(density)})', (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, f'Speed: {avg_speed:.2f} px/frame', (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if total_count > crowd_limit:
            cv2.putText(overlay, 'Crowd Limit Exceeded!', (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Write frame
        out.write(overlay)

        # Update previous frame
        prev_frame_gray = frame_gray.copy()

    cap.release()
    out.release()
    return crowd_limit_exceeded, output_path, max_density, max_count


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password are required'}), 400
    
    # Check if user exists in MongoDB
    user = mongo.db.users.find_one({'email': email})
    
    if user and check_password_hash(user['password'], password):
        return jsonify({
            'success': True, 
            'message': 'Login successful',
            'user': {
                'id': str(user['_id']),
                'email': user['email']
            }
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirmPassword')
    
    if not email or not password or not confirm_password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    if password != confirm_password:
        return jsonify({'success': False, 'message': 'Passwords do not match'}), 400
    
    # Check if user already exists
    if mongo.db.users.find_one({'email': email}):
        return jsonify({'success': False, 'message': 'Email already registered'}), 400
    
    # Create new user with hashed password
    user_id = mongo.db.users.insert_one({
        'email': email,
        'password': generate_password_hash(password),
        'created_at': datetime.now(),
        'last_login': None,
        'analysis_history': []
    }).inserted_id
    
    return jsonify({
        'success': True, 
        'message': 'Registration successful',
        'user': {
            'id': str(user_id),
            'email': email
        }
    })

@app.route('/api/save_analysis', methods=['POST'])
def save_analysis():
    data = request.get_json()
    user_id = data.get('user_id')
    analysis_data = data.get('analysis_data')
    
    if not user_id or not analysis_data:
        return jsonify({'success': False, 'message': 'Missing required data'}), 400
    
    try:
        # Add timestamp to analysis data
        analysis_data['timestamp'] = datetime.now()
        
        # Save to MongoDB
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$push': {'analysis_history': analysis_data}}
        )
        
        return jsonify({'success': True, 'message': 'Analysis saved successfully'})
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
        return jsonify({'success': False, 'message': 'Error saving analysis'}), 500

@app.route('/api/get_analysis_history', methods=['GET'])
def get_analysis_history():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400
    
    try:
        user = mongo.db.users.find_one(
            {'_id': ObjectId(user_id)},
            {'analysis_history': 1}
        )
        
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404
        
        # Convert ObjectId to string for JSON serialization
        history = user.get('analysis_history', [])
        for item in history:
            if '_id' in item:
                item['_id'] = str(item['_id'])
        
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        logger.error(f"Error fetching analysis history: {e}")
        return jsonify({'success': False, 'message': 'Error fetching history'}), 500

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_realtime_data')
def get_realtime_data():
    with real_time_data['lock']:
        data = {
            'timestamps': real_time_data['timestamps'],
            'density': real_time_data['density_values'],
            'count': real_time_data['count_values'],
            'speed': real_time_data['speed_values']
        }
    return jsonify(data)


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        is_webcam = request.form.get('isWebcam') == 'true'
        
        if is_webcam:
            # Handle webcam frame analysis
            if 'frame' not in request.files:
                return jsonify({"error": "No frame provided"}), 400
                
            frame_file = request.files['frame']
            email = request.form.get('email')
            crowd_limit_str = request.form.get('crowdLimit', '0')
            whatsapp_number = request.form.get('whatsappNumber')
            

             # Convert crowd limit to integer with default value
            try:
                crowd_limit = int(crowd_limit_str) if crowd_limit_str else 0
            except ValueError:
                crowd_limit = 0
            # Read and process the frame
            frame = cv2.imdecode(np.frombuffer(frame_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"success": False, "error": "Could not decode frame"}), 400
            result = process_single_frame(frame, crowd_limit)
            
            if not result['success']:
                return jsonify({"success": False, "error": result.get('error', 'Unknown error')}), 500
            
            # Send alerts if crowd limit exceeded
            if result['crowd_limit_exceeded']:
                alert_message = f"Crowd limit ({crowd_limit}) exceeded! Current count: {result['count']}"
                if email:
                    send_email(email, "Crowd Limit Exceeded", alert_message)
                if whatsapp_number:
                    send_whatsapp_alert(whatsapp_number, alert_message)
            # Convert annotated frame to base64 for sending back
            _, buffer = cv2.imencode('.jpg', result['annotated_frame'])
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                "success": True,
                "count": result['count'],
                "density": result['density'],
                "speed": 0.0,  # Speed calculation not available for single frames
                "crowd_limit_exceeded": result['crowd_limit_exceeded'],
                "message": "Frame processed successfully"
            })
        else:
            # Handle video file upload
            if 'videoFile' not in request.files or 'email' not in request.form or 'crowdLimit' not in request.form:
                return jsonify({"error": "No file, email or crowd limit provided"}), 400

            file = request.files['videoFile']
            email = request.form['email']
            crowd_limit = int(request.form['crowdLimit'])
            whatsapp_number = request.form.get('whatsappNumber')

            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({"error": "Invalid file"}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            crowd_limit_exceeded, processed_video_path, max_density, max_count = process_video_with_density(filepath, crowd_limit)

            if crowd_limit_exceeded:
                alert_message = f"Crowd limit ({crowd_limit}) exceeded! Max density: {max_density:.2f}, Max count: {max_count}"
                send_email(email, "Crowd Limit Exceeded", alert_message)
                if whatsapp_number:
                    send_whatsapp_alert(whatsapp_number, alert_message)

            return jsonify({
                "success": True,
                "crowd_limit_exceeded": crowd_limit_exceeded,
                "processed_video_url": f"/uploads/processed/{os.path.basename(processed_video_path)}",
                "max_density": max_density,
                "max_count": max_count,
                "message": "Video processed successfully"
            })
            
    except Exception as e:
        logger.error(f"Error processing video/frame: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

'''if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)'''
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)