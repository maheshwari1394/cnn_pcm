Real-Time Fraud Call Detection using CNN

This project implements a real-time fraud call detection system that leverages Convolutional Neural Networks (CNNs) to analyze live audio conversations and classify them as Fraudulent or Normal. The solution continuously captures microphone input, applies a sliding window mechanism, transforms audio into mel-spectrograms, and performs inference using a fine-tuned MobileNetV2 deep learning model.

Key Features

🎙️ Live Audio Capture
Captures microphone audio in real time using the Web Audio API.

🔄 Sliding Window Processing
Analyzes audio using overlapping windows (3-second window with 1.5-second stride) to ensure continuous monitoring.

📊 Spectrogram Transformation
Converts audio segments into mel-spectrogram images suitable for CNN-based classification.

🤖 CNN-Based Fraud Detection
Uses a fine-tuned MobileNetV2 model to predict fraud likelihood.

📈 Live Prediction Updates
Displays chunk-level predictions and confidence scores during the call.

✅ Conversation-Level Decision
Aggregates all chunk predictions to produce a final classification for the entire conversation.

💬 Optional Speech Transcription
Supports speech-to-text via the Web Speech API (browser-dependent).

🎨 Interactive User Interface
Modern, responsive UI with real-time updates and status indicators.

System Architecture

The application follows a client–server architecture consisting of:

Frontend (Client)

HTML and JavaScript-based web interface

Uses AudioWorklet API for low-latency audio processing

Communicates with backend through RESTful APIs

Backend (Server)

Flask-based REST API

TensorFlow/Keras model for inference

Audio preprocessing and spectrogram generation

In-memory session handling for conversations

PROJECT STRUCTURE
laptop-laptop-cnn-github/
├── client_mod/
│   ├── index.html          # Main web interface
│   └── audio_processor.js  # AudioWorklet processor for real-time audio capture
├── server_mod/
│   ├── app.py              # Flask API server
│   ├── text_infer_cnn.py   # CNN inference and spectrogram generation
│   ├── mobilenetv2_fraud_detector_final_focal_6domains.keras  # Trained model (may not be in repo)
│   ├── temp_audios/        # Temporary audio files (auto-created, gitignored)
│   └── temp_spectrograms/  # Temporary spectrogram images (auto-created, gitignored)
├── client_serve_appmod.py  # Flask server for serving static client files
├── requirements_mod.txt    # Python dependencies
└── README.md               # This file
