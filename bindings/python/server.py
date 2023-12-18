from flask import Flask, request, jsonify
from whisper_cpp import WhisperCpp
import numpy as np
import time

app = Flask(__name__)
audio_model = WhisperCpp(model="/home/artem/git/whisper.cpp/models/ggml-large-v3-q5_0.bin", use_gpu=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    start = time.time()
    try:
        audio_data = request.files['audio_data'].read()
        language = request.files['language'].read().decode()
        print(f"received {len(audio_data)}")
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        transcription = audio_model.transcribe(audio_np, language=language)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return "", time.time() - start
    else:
        print(f"[{len(audio_data) / 16000:3.1f}s, {time.time() - start:3.1f}s] {transcription}")
        return jsonify({"transcription": transcription, "server_time": time.time() - start})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)