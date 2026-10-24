from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_VIDEO_LENGTH = 30  # seconds

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Simplified CNN model for demonstration
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeDetector().to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_frames(video_path, num_frames=10):
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly throughout the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def generate_attention_map(image_tensor, prediction):
    """Generate attention/explanation map using gradient-based approach"""
    # Simplified attention map generation
    # In production, use GradCAM or similar techniques
    with torch.enable_grad():
        image_tensor.requires_grad = True
        output = model(image_tensor)
        
        # Get the score for the predicted class
        score = output[0, prediction.argmax()]
        
        # Backward pass to get gradients
        score.backward()
        
        # Get gradients
        gradients = image_tensor.grad
        
        # Create attention map from gradients
        attention = gradients.abs().mean(dim=1).squeeze().cpu().detach().numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    return attention


def analyze_video(video_path):
    """Analyze video for deepfake detection"""
    frames = extract_frames(video_path, num_frames=10)
    
    if len(frames) == 0:
        return {
            'error': 'Could not extract frames from video'
        }
    
    predictions = []
    attention_maps = []
    frame_results = []
    
    for i, frame in enumerate(frames):
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Transform
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            deepfake_prob = probs[0, 1].item()
            real_prob = probs[0, 0].item()
        
        is_deepfake = deepfake_prob > 0.5
        
        # Generate attention map
        attention_map = generate_attention_map(input_tensor, probs)
        
        # Convert attention map to base64 for display
        attention_pil = Image.fromarray((attention_map * 255).astype(np.uint8))
        attention_pil = attention_pil.resize((224, 224))
        buffered = io.BytesIO()
        attention_pil.save(buffered, format='PNG')
        attention_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        frame_results.append({
            'frame_number': i,
            'is_deepfake': is_deepfake,
            'deepfake_probability': deepfake_prob,
            'real_probability': real_prob,
            'attention_map': f'data:image/png;base64,{attention_base64}'
        })
        
        predictions.append(deepfake_prob)
    
    # Overall video prediction
    avg_deepfake_prob = np.mean(predictions)
    overall_result = {
        'is_deepfake': avg_deepfake_prob > 0.5,
        'confidence': avg_deepfake_prob if avg_deepfake_prob > 0.5 else 1 - avg_deepfake_prob,
        'deepfake_probability': float(avg_deepfake_prob),
        'real_probability': float(1 - avg_deepfake_prob),
        'frame_analysis': frame_results,
        'total_frames_analyzed': len(frames)
    }
    
    return overall_result


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': str(device)})


@app.route('/api/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze video
        result = analyze_video(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/frame', methods=['POST'])
def detect_frame():
    """Detect deepfake in a single frame"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    try:
        # Read image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            deepfake_prob = probs[0, 1].item()
            real_prob = probs[0, 0].item()
        
        is_deepfake = deepfake_prob > 0.5
        
        # Generate attention map
        attention_map = generate_attention_map(input_tensor, probs)
        
        # Convert attention map to base64
        attention_pil = Image.fromarray((attention_map * 255).astype(np.uint8))
        attention_pil = attention_pil.resize((224, 224))
        buffered = io.BytesIO()
        attention_pil.save(buffered, format='PNG')
        attention_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        result = {
            'is_deepfake': is_deepfake,
            'confidence': deepfake_prob if is_deepfake else real_prob,
            'deepfake_probability': deepfake_prob,
            'real_probability': real_prob,
            'attention_map': f'data:image/png;base64,{attention_base64}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
