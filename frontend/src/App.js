import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      
      // Create video preview
      const videoUrl = URL.createObjectURL(selectedFile);
      setPreview(videoUrl);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a video file first');
      return;
    }

    setAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#ef4444';
    if (confidence > 0.6) return '#f97316';
    if (confidence > 0.4) return '#eab308';
    return '#22c55e';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎭 AI Deepfake Detector</h1>
        <p>Upload a video to detect deepfakes with explainable AI</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <div className="file-input-wrapper">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              id="video-upload"
              className="file-input"
            />
            <label htmlFor="video-upload" className="file-label">
              {file ? file.name : 'Choose Video File'}
            </label>
          </div>

          {preview && (
            <div className="preview-section">
              <video
                src={preview}
                controls
                className="video-preview"
                width="400"
              />
              <button
                onClick={handleAnalyze}
                disabled={analyzing}
                className="analyze-button"
              >
                {analyzing ? 'Analyzing...' : 'Analyze Video'}
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            <p>⚠️ {error}</p>
          </div>
        )}

        {result && (
          <div className="results-section">
            <div className="overall-result">
              <h2>Overall Result</h2>
              <div
                className={`result-badge ${result.is_deepfake ? 'deepfake' : 'real'}`}
              >
                {result.is_deepfake ? '🚨 DEEPFAKE DETECTED' : '✅ LIKELY REAL'}
              </div>
              <div className="confidence-meter">
                <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${result.confidence * 100}%`,
                      backgroundColor: getConfidenceColor(result.confidence),
                    }}
                  />
                </div>
              </div>
              <div className="probabilities">
                <p>Deepfake Probability: {(result.deepfake_probability * 100).toFixed(1)}%</p>
                <p>Real Probability: {(result.real_probability * 100).toFixed(1)}%</p>
              </div>
            </div>

            <div className="frame-analysis">
              <h3>Frame-by-Frame Analysis ({result.total_frames_analyzed} frames)</h3>
              <div className="frames-grid">
                {result.frame_analysis.map((frame) => (
                  <div key={frame.frame_number} className="frame-card">
                    <p className="frame-number">Frame {frame.frame_number}</p>
                    <div
                      className={`frame-status ${frame.is_deepfake ? 'deepfake' : 'real'}`}
                    >
                      {frame.is_deepfake ? '🚨' : '✅'}
                    </div>
                    <p className="frame-prob">
                      Deepfake: {(frame.deepfake_probability * 100).toFixed(1)}%
                    </p>
                    <div className="attention-map">
                      <p>Attention Map:</p>
                      <img
                        src={frame.attention_map}
                        alt={`Attention map for frame ${frame.frame_number}`}
                        className="attention-image"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by AI & PyTorch | For educational and research purposes</p>
      </footer>
    </div>
  );
}

export default App;
