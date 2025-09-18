import matplotlib
matplotlib.use('Agg')  # headless server-friendly backend
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# Directories & constants
basedir = os.path.abspath(os.path.dirname(__file__))
SPEC_DIR = Path(basedir) / 'temp_spectrograms'
MODEL_PATH = Path(basedir) / 'mobilenetv2_fraud_detector_final_focal_6domains.keras'

os.makedirs(SPEC_DIR, exist_ok=True)

SAMPLE_RATE = 22050
IMG_SIZE = (224, 224)

# Load model once globally
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Keras model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
    model = None


def generate_spectrogram_from_array(audio_array: np.ndarray, sr: int, out_path: str) -> bool:
    try:
        y = np.array(audio_array, dtype=np.float32)

        # --- Sanitize ---
        if not np.all(np.isfinite(y)):
            print("[WARN] Audio contains non-finite values, fixing...")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Normalize amplitude ---
        if len(y) > 0:
            max_val = np.max(np.abs(y))
            if max_val > 1.0:
                y = y / (max_val + 1e-9)
                print(f"[INFO] Normalized audio, new max={np.max(np.abs(y)):.4f}")

        # --- Pad to at least 1 second ---
        min_samples = int(sr * 1.0)
        print(f"[DEBUG] min_samples={min_samples}, current_len={len(y)}")
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), 'constant')

        # --- Generate spectrogram ---
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        if not np.all(np.isfinite(S_dB)) or S_dB.size == 0:
            print(f"[ERROR] Spectrogram data invalid")
            return False

        # --- Plot and save ---
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, format='%+2.0f dB', ax=ax)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(str(out_path), dpi=72, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"[INFO] Spectrogram saved to {out_path}")
        return True

    except Exception as e:
        print(f"[ERROR] generate_spectrogram_from_array failed: {e}")
        return False


def predict_image(img_path: str):
    try:
        if model is None:
            print("[ERROR] Model not loaded.")
            return None, None

        img = image.load_img(str(img_path), target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array, verbose=0)[0]

        # Handle binary classification (softmax or sigmoid)
        if preds.shape and len(preds) == 2:
            fraud_prob = float(preds[1])  # second neuron → fraud class
        else:
            fraud_prob = float(preds[0])  # single output

        # Decision threshold at 0.60
        label = "Fraud" if fraud_prob < 0.60 else "Normal"
        print(f"[INFO] Prediction: {label} (Fraud prob: {fraud_prob:.4f})")

        return label, fraud_prob

    except Exception as e:
        print(f"[ERROR] predict_image failed for {img_path}: {e}")
        return None, None
