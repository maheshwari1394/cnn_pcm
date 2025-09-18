# client_serve_app.py
from flask import Flask, send_from_directory
from flask_cors import CORS
import os

basedir = os.path.abspath(os.path.dirname(__file__))
client_dir = os.path.join(basedir, 'client_mod')  # put index.html + audio_processor.js here

app = Flask(__name__, static_folder=client_dir)
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
