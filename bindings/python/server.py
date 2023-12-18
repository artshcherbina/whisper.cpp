from flask import Flask, request
from whisper_cpp import WhisperCpp
import numpy as np

app = Flask(__name__)
audio_model = WhisperCpp(model="/home/artem/git/whisper.cpp/models/ggml-large-v3-q5_0.bin", use_gpu=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_data = request.data
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    transcription = audio_model.transcribe(audio_np, language="en").strip()
    return transcription

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)