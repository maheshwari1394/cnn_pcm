# Fraud Call Detection using CNN

A real-time fraud call detection system that uses Convolutional Neural Networks (CNN) to analyze audio conversations and classify them as **Normal** or **Fraud**. The system processes audio in real time using a sliding window approach, converts audio to spectrograms, and uses a MobileNetV2-based deep learning model for prediction.

---

## Features

🎧 **Live Audio Capture**  
Records and processes microphone input in real time using browser-based audio APIs

🔁 **Overlapping Window Processing**  
Continuously analyzes audio using a 3-second frame with a 1.5-second stride

🖼️ **Audio-to-Image Conversion**  
Transforms raw audio signals into mel-spectrogram representations for model input

🧠 **CNN-Based Detection Model**  
Utilizes a fine-tuned MobileNetV2 convolutional network for fraud classification

⚡ **Continuous Prediction Output**  
Generates confidence scores and predictions for each analyzed audio segment

✔️ **Conversation-Level Decision**  
Combines segment-wise predictions to determine the final call classification

🗣️ **Optional Speech Recognition**  
Supports speech-to-text transcription through the Web Speech API

🖥️ **Interactive User Interface**  
Responsive and modern UI with real-time visual feedback and status indicators

---

## Notes

This folder contains a fully working **CNN laptop–laptop PCM audio pipeline**.

- `temp_audios/` and `temp_spectrograms/` folders are created automatically at runtime and should be excluded from version control.
- The trained `.keras` model file may be omitted from the repository due to size constraints and should be placed manually if required.

---

## Working

### 1. Client-Side Processing (Frontend)

- The user accesses the web interface through `index.html`, which initializes microphone access and loads a custom AudioWorklet processor.
- When recording starts, the browser requests microphone permission and establishes a new conversation session by calling the backend `/start_conversation` API.
- A unique `session_id` returned by the server is used to associate all audio chunks with the same conversation.
- Audio is captured as raw PCM data and continuously buffered inside the AudioWorklet node.
- Once the buffer reaches the defined window size, the audio segment is extracted and sent to the main JavaScript thread.
- Each audio chunk is transmitted to the backend using the `/process_audio` endpoint along with the session identifier.
- Recording continues until the user stops the microphone, after which a final audio chunk is sent with a flag indicating the end of the conversation.

---

### 2. Server-Side Processing (Backend)

- The backend maintains an in-memory session store to track incoming audio chunks for each active conversation.
- All received audio segments are appended sequentially without triggering inference during recording.
- When the final chunk indicator is received, the server consolidates the entire audio stream for that session.
- The combined audio signal is converted into a mel-spectrogram image using predefined audio processing parameters.
- The spectrogram is resized and normalized to match the CNN input format.
- The processed image is passed to a fine-tuned MobileNetV2-based CNN model for classification.
- The model outputs a fraud probability score, which is mapped to either **Fraud** or **Normal** based on the confidence threshold.
- After inference, temporary files and session data are removed to free system resources.

---

### 3. Output and Feedback

- The final prediction and confidence score are returned to the client as a JSON response.
- The web interface displays real-time status updates during recording and shows the final classification once analysis is complete.

---

## Prediction Threshold

The system uses a confidence threshold of **0.60**:

- Confidence < 0.60 → **Fraud**
- Confidence ≥ 0.60 → **Normal**

