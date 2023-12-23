#! python3.7

import argparse
import os
import re
import time
import pyperclip
import pygame
import numpy as np
import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

from scipy.io import wavfile
from whisper_cpp import WhisperCpp
import json
from playsound import playsound
from pynput import keyboard
import threading
import requests

pygame.mixer.init()

class KeyListener:
    def __init__(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.key_callbacks = {}  # Key combinations and their callbacks
        self.pressed_keys = set()  # Currently pressed keys
        self.upper_case = True

    def register_key(self, key: str, callback, arguments=None):  # Register a key combination and its callback
        key = '+'.join(k.lower() if len(k) > 1 else k for k in key.split('+'))  # Add 'Key.' prefix to special keys
        self.key_callbacks[key] = callback, arguments

    def on_press(self, key):
        key = str(key).replace("Key.", "").replace("'", "")
        # print(f"pressed {key}")
        self.pressed_keys.add(key)
        for key_combination, (callback, arguments) in self.key_callbacks.items():
            keys = set(key_combination.split('+'))
            # print(keys, self.pressed_keys, keys.issubset(self.pressed_keys))
            if keys.issubset(self.pressed_keys):
                callback(*arguments)

    def on_release(self, key):
        try:
            key = str(key).replace("Key.", "").replace("'", "")
            # print(f"released {key}")
            self.pressed_keys.remove(key)
        except Exception as e:
            print(f"Exception during on_release: {e}")

    def start(self):
        self.listener.run()
        # thread = threading.Thread(target=self.listener.start)
        # thread.start()


class Transcriber:
    def __init__(self):
        self.current_file_directory = os.path.dirname(os.path.abspath(__file__))
        with open(self.current_file_directory + '/config.json', 'r') as f:
            self.args = argparse.Namespace(**json.load(f))

        self.data_queue = Queue()
        self.recorder = sr.Recognizer(self.args.energy_threshold, self.args.pause_timeout)
        # self.recorder.dynamic_energy_threshold = False
        pygame.mixer.music.set_volume(self.args.volume)

        if 'linux' in platform:
            mic_name = self.args.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            self.source = sr.Microphone(sample_rate=16000)

        if self.args.server is None:
            self.audio_model = WhisperCpp(model="/home/artem/git/whisper.cpp/models/ggml-large-v3-q5_0.bin", use_gpu=self.args.use_gpu)
            print("Model loaded.\n")

        # self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

        self.key_listener = KeyListener()
        for language, key in self.args.languages.items():
            self.key_listener.register_key(key, self.run, (language, ))

    def start(self):
        self.key_listener.start()

    def play_sound(self, enabled):
        pygame.mixer.music.load(f"sounds/done.wav")
        pygame.mixer.music.play()

    def run(self, language):
        print(f"starting recording for {language}...")
        with  self.source as s:
            audio_data = self.recorder.listen(s, None, phrase_time_limit=self.args.record_timeout).frame_data
        start = time.time()
        if self.args.server:
            print(f"sending {len(audio_data)}")
            try:
                response = requests.post(self.args.server, files={"audio_data": audio_data, "language": language.encode()})
                data = response.json()
                original_text = data['transcription']
                server_time = data['server_time']
            except Exception as e:
                print(f"transribtion failed: {e}")
                return
        else:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            original_text = self.audio_model.transcribe(audio_np, language=language)
            server_time = 0
        text = original_text
        print(f"original_text: {original_text}")

        text = text.rstrip('.').strip()

        text = (text[:1].upper() if self.key_listener.upper_case else text[:1].lower()) + text[1:]

        # Replace punctuations
        for punctuation, parts in self.args.punctuations.items():
            for part in parts:
                if part in text.lower():
                    text = re.sub(" *" + part + "\S*", punctuation, text, flags=re.IGNORECASE)

        text = re.sub(r'\s([:.])', r'\1', text)

        total_time = time.time() - start
        print(f"[{len(audio_data) / 16000:3.1f}s + {server_time:3.1f}s + {total_time - server_time:3.1f}s] <{original_text}> -> <{text}>")
        if text:
            self.play_sound(True)
            pyperclip.copy(text)

if __name__ == "__main__":
    transcriber = Transcriber()
    transcriber.start()
