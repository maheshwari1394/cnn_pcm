Fraud Call Detection using CNN
A real-time fraud call detection system that uses Convolutional Neural Networks (CNN) to analyze audio conversations and classify them as "Normal" or "Fraud". The system processes audio in real-time using a sliding window approach, converts audio to spectrograms, and uses a MobileNetV2-based deep learning model for prediction.

Features
🎧 Live Audio Capture: Records and processes microphone input in real time using browser-based audio APIs
🔁 Overlapping Window Processing: Continuously analyzes audio using a 3-second frame with a 1.5-second stride
🖼️ Audio-to-Image Conversion: Transforms raw audio signals into mel-spectrogram representations for model input
🧠 CNN-Based Detection Model: Utilizes a fine-tuned MobileNetV2 convolutional network for fraud classification
⚡ Continuous Prediction Output: Generates confidence scores and predictions for each analyzed audio segment
✔️ Conversation-Level Decision: Combines segment-wise predictions to determine the final call classification
🗣️ Optional Speech Recognition: Supports speech-to-text transcription through the Web Speech API
🖥️ Interactive User Interface: Responsive and modern UI with real-time visual feedback and status indicators
Note: This folder contains fully working cnn laptop-laptop PCM audio pipeline

Note:

The temp_audios/ and temp_spectrograms/ folders are created automatically at runtime and should be excluded from version control.
The trained .keras model file may be omitted from the repository due to size constraints and should be placed manually if required.
WORKING
1. Client-Side Processing (Frontend)
The user accesses the web interface through index.html, which initializes microphone access and loads a custom AudioWorklet processor.
When recording starts, the browser requests microphone permission and establishes a new conversation session by calling the backend /start_conversation API.
A unique session_id returned by the server is used to associate all audio chunks with the same conversation.
Audio is captured as raw PCM data and continuously buffered inside the AudioWorklet node.
Once the buffer reaches the defined window size, the audio segment is extracted and sent to the main JavaScript thread.
Each audio chunk is transmitted to the backend using the /process_audio endpoint along with the session identifier.
Recording continues until the user stops the microphone, after which a final audio chunk is sent with a flag indicating the end of the conversation.
2. Server-Side Processing (Backend)
The backend maintains an in-memory session store to track incoming audio chunks for each active conversation.
All received audio segments are appended sequentially without triggering inference during recording.
When the final chunk indicator is received, the server consolidates the entire audio stream for that session.
The combined audio signal is converted into a mel-spectrogram image using predefined audio processing parameters.
The spectrogram is resized and normalized to match the CNN input format.
The processed image is passed to a fine-tuned MobileNetV2-based CNN model for classification.
The model outputs a fraud probability score, which is mapped to either Fraud or Normal based on the confidence threshold.
After inference, temporary files and session data are removed to free system resources.
3. Output and Feedback
The final prediction and confidence score are returned to the client as a JSON response.
The web interface displays real-time status updates during recording and shows the final classification once analysis is complete.
Prediction Threshold
The system uses a confidence threshold of 0.60:

Confidence < 0.60: Classified as "Fraud"
Confidence ≥ 0.60: Classified as "Normal"
API Endpoints
POST /start_conversation
Initializes a new conversation session.

Response:

{
  "session_id": "uuid-string"
}
POST /process_audio
Processes an audio chunk and returns prediction.

Request:

session_id (form-data): Session ID from start_conversation
audio (file): Audio file (WAV format)
is_last_chunk (optional, form-data): "true" if this is the final chunk
Response (intermediate chunk):

{
  "prediction": "Normal" | "Fraud",
  "confidence": 0.80
}
Response (final chunk):

{
  "final_prediction": "Normal" | "Fraud",
  "average_confidence": 0.78,
  "is_final": true
}
Configuration
Audio Processing Parameters
In client_mod/index.html, you can adjust:

WINDOW_DURATION_SECONDS: Duration of each analysis window (default: 3 seconds)
HOP_DURATION_SECONDS: Time between window starts (default: 1.5 seconds)
Server Configuration
In server_mod/app.py:

TEMP_AUDIO_DIR: Directory for temporary audio files
TEMP_SPECTROGRAM_DIR: Directory for temporary spectrogram images
Server port: Default is 4567 (configurable in app.run())
In server_mod/text_infer_cnn.py:

SAMPLE_RATE: Audio sample rate (default: 22050 Hz)
IMG_SIZE: Spectrogram image size (default: 224x224)
Confidence threshold: 0.60 (hardcoded in prediction logic)
How It Works
Audio Capture: The browser captures audio from the microphone using the AudioWorklet API
Sliding Window: Audio is divided into overlapping windows (3s window, 1.5s hop)
Spectrogram Conversion: Each audio chunk is converted to a mel-spectrogram image
CNN Prediction: The MobileNetV2 model processes the spectrogram and outputs a confidence score
Aggregation: Individual predictions are averaged to determine the final conversation classification
Real-time Display: Predictions are displayed in real-time in the web interface
Troubleshooting
Browser Issues
AudioWorklet not loading: Ensure you're accessing the page via http://localhost:5000 (not file://)
Microphone permissions denied: Check browser settings and allow microphone access
Speech recognition not working: Some browsers may not support Web Speech API; this is optional
Server Issues
Model file not found: Ensure mobilenetv2_fraud_detector_final_focal_6domains.keras exists in server_mod/
Port already in use: Change the port in app.py or client_serve_appmod.py if 4567 or 5000 are occupied
CORS errors: Ensure Flask-Cors is installed and CORS is enabled in app.py
Performance Issues
Slow predictions: The model inference may take time; consider using GPU acceleration with TensorFlow
Memory issues: Temporary files are cleaned up automatically, but monitor server memory usage
Technical Details
Model Architecture
Base: MobileNetV2 (pre-trained)
Output: Binary classification (Normal/Fraud)
Input: 224x224 RGB spectrogram images
Activation: Sigmoid (confidence score)
Audio Processing
Sample Rate: 22050 Hz
Format: WAV (16-bit PCM)
Channels: Mono
Spectrogram Type: Mel-spectrogram (power-to-dB conversion)
Limitations
Requires an active internet connection for browser features (if using online speech recognition)
Model accuracy depends on training data quality
Real-time processing may have latency depending on system resources
Session data is stored in-memory (not persistent across server restarts)
