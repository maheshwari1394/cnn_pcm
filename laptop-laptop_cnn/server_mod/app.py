import os
import uuid
import numpy as np
import threading
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# import your inference module
import text_infer_cnn as inference_module

app = Flask(__name__)
CORS(app)

# In-memory sessions store
sessions = {}

TEMP_SPECTROGRAM_DIR = 'temp_spectrograms'
os.makedirs(TEMP_SPECTROGRAM_DIR, exist_ok=True)


@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'lock': threading.Lock(),
        'audio_buffer': np.zeros(0, dtype=np.float32),
        'samplerate': None
    }
    print(f"[INFO] New session {session_id}")
    return jsonify({'session_id': session_id}), 200


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # get session id
        session_id = request.form.get('session_id') or request.headers.get('X-Session-Id')
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Invalid or expired session ID."}), 410

        # detect last chunk
        is_last_header = request.headers.get('X-Is-Last-Chunk', 'false').lower() == 'true'
        is_last_form = request.form.get('is_last_chunk', 'false').lower() == 'true'
        is_last = is_last_header or is_last_form

        # now we expect raw PCM (float32) bytes
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # read the raw PCM Float32 directly
        raw_bytes = audio_file.read()
        pcm_array = np.frombuffer(raw_bytes, dtype=np.float32)

        # Debug: check stats of this chunk
        if pcm_array.size > 0:
            print(f"[DEBUG] Received chunk stats: len={len(pcm_array)}, "
                  f"min={pcm_array.min()}, max={pcm_array.max()}, "
                  f"any_nan={np.isnan(pcm_array).any()}, "
                  f"any_inf={np.isinf(pcm_array).any()}")
        else:
            print("[WARN] Empty PCM chunk received!")

        sess = sessions[session_id]
        with sess['lock']:
            # sample rate provided? (from client via header)
            sr = request.headers.get('X-Sample-Rate')
            if sr is not None:
                sr = int(sr)
            else:
                sr = sess['samplerate'] or 22050  # fallback

            if sess['samplerate'] is None:
                sess['samplerate'] = sr
            elif sess['samplerate'] != sr:
                print(f"[WARN] sample rate mismatch: existing {sess['samplerate']} vs chunk {sr}. Using chunk rate.")
                sess['samplerate'] = sr

            # append new PCM samples
            sess['audio_buffer'] = np.concatenate([sess['audio_buffer'], pcm_array])

        print(f"[INFO] Received chunk for session {session_id} (len={len(pcm_array)} samples). "
              f"Total buffered={len(sess['audio_buffer'])}. is_last={is_last}")

        if not is_last:
            return jsonify({"received": True, "is_final": False}), 200

        # last chunk: run inference
        with sess['lock']:
            full_audio = sess['audio_buffer'].copy()
            sr = sess['samplerate'] or 22050

        unique_id = uuid.uuid4().hex[:8]
        temp_spec_path = os.path.join(TEMP_SPECTROGRAM_DIR, f"spec_{session_id}_{unique_id}.png")

        # generate spectrogram from array
        ok = inference_module.generate_spectrogram_from_array(full_audio, sr, temp_spec_path)
        if not ok:
            return jsonify({"error": "Failed to generate spectrogram"}), 500

        label, confidence = inference_module.predict_image(temp_spec_path)
        if label is None:
            return jsonify({"error": "Prediction failed"}), 500

        avg_conf = float(confidence)
        final_prediction = "Fraud" if avg_conf < 0.60 else "Normal"

        del sessions[session_id]
        print(f"[INFO] Final result for session {session_id}: {final_prediction} ({avg_conf:.4f}). Session removed.")

        return jsonify({
            "final_prediction": final_prediction,
            "average_confidence": avg_conf,
            "is_final": True,
            "spectrogram_path": temp_spec_path
        }), 200

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)