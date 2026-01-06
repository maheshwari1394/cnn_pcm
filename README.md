Laptop Client – Laptop Server (raw PCM)

Overview
This project implements a client–server voice anomaly detection system where audio is captured on a client laptop (browser) and analyzed on a server laptop using a deep learning model.

Pipeline Summary

Browser captures microphone audio in raw PCM format
Audio is buffered and sent to the server in chunks
Server aggregates the full conversation
Final analysis is performed after recording ends
CNN model classifies audio as Normal or Fraud

Key Features

Real-time PCM audio capture
Chunk-based client-server communication
End-of-conversation ML inference
Spectrogram-based CNN prediction
