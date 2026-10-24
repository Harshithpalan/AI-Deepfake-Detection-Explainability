# AI Deepfake Detection + Explainability

A web application for detecting deepfakes in videos with explainable AI features including attention maps and frame-by-frame analysis.

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: React
- **ML Framework**: PyTorch
- **Computer Vision**: OpenCV, Pillow

## Features

- 🎥 Video upload and analysis
- 🎯 Deepfake detection with confidence scores
- 📊 Frame-by-frame analysis
- 🔍 Attention/explanation maps for each frame
- 🎨 Modern, responsive UI
- ⚡ Real-time processing

## Prerequisites

- Python 3.8+
- Node.js 16+
- pip
- npm

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Click "Choose Video File" to upload a video (mp4, avi, mov, mkv)
3. Preview the video
4. Click "Analyze Video" to start detection
5. View the results:
   - Overall deepfake probability
   - Confidence score
   - Frame-by-frame analysis
   - Attention maps showing which regions the model focused on

## API Endpoints

### Health Check
```
GET /api/health
```

### Detect Deepfake in Video
```
POST /api/detect
Content-Type: multipart/form-data

Body:
- video: video file
```

### Detect Deepfake in Single Frame
```
POST /api/detect/frame
Content-Type: multipart/form-data

Body:
- image: image file
```

## Model Architecture

The application uses a simplified CNN model for demonstration:
- 3 convolutional layers with ReLU activation
- Max pooling layers
- Fully connected layers with dropout
- Binary classification (real vs deepfake)

## Explainability Features

- **Attention Maps**: Gradient-based visualization showing which regions influenced the model's decision
- **Frame-by-Frame Analysis**: Individual analysis of sampled frames from the video
- **Confidence Scores**: Probability estimates for both real and deepfake predictions

## Notes

- The current model is a simplified version for demonstration purposes
- For production use, consider:
  - Training on large deepfake datasets (FaceForensics++, DFDC, etc.)
  - Using pre-trained models (Xception, EfficientNet, etc.)
  - Implementing more sophisticated explainability techniques (GradCAM, Integrated Gradients)
  - Adding temporal analysis for video consistency
  - Implementing model quantization for faster inference

## Future Enhancements

- [ ] Add real-time webcam detection
- [ ] Support for live video streams
- [ ] Multiple model ensemble
- [ ] Batch processing
- [ ] User authentication
- [ ] History and saved analyses
- [ ] Export reports
- [ ] Audio deepfake detection

## License

This project is for educational and research purposes only.

## Disclaimer

This tool is provided for educational and research purposes. The results should not be used as definitive evidence in any legal or official context. Always verify results through multiple methods and expert analysis.
