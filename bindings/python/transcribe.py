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

pygame.mixer.init()

class KeyListener:
    def __init__(self, recognition_callback):
        self.last_shift_press = 0
        self.shift_pressed = False
        self.recognition = True
        self.recognition_callback = recognition_callback
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        if key == keyboard.Key.shift:
            self.shift_pressed = True
            # print('Shift key was pressed')
        if key == keyboard.Key.f12:
            self.recognition = not self.recognition
            # print('Recognition ' + ("enabled" if self.recognition else "disabled"))
            self.recognition_callback(self.recognition)

    def on_release(self, key):
        if key == keyboard.Key.shift and self.shift_pressed:
            self.shift_pressed = False
            self.last_shift_press = time.time()
            # print('Shift key was released, it was last pressed at', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_shift_press)))

    def start(self):
        thread = threading.Thread(target=self.listener.start)
        thread.start()

class Transcriber:
    def __init__(self):
        self.current_file_directory = os.path.dirname(os.path.abspath(__file__))
        with open(self.current_file_directory + '/config.json', 'r') as f:
            self.args = argparse.Namespace(**json.load(f))

        self.data_queue = Queue()
        self.recorder = sr.Recognizer(self.args.energy_threshold, self.args.pause_timeout)
        self.recorder.dynamic_energy_threshold = False
        self.language = "en"
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

        self.audio_model = WhisperCpp(model="/home/artem/git/whisper.cpp/models/ggml-large-v3-q5_0.bin", use_gpu=self.args.use_gpu)
        self.record_timeout = self.args.record_timeout

        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

        print("Model loaded.\n")

        self.key_listener = KeyListener(self.recognition_callback)
        self.key_listener.start()

    def record_callback(self, _, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def recognition_callback(self, enabled):
        pygame.mixer.music.load(f"{self.current_file_directory}/sounds/{self.language}_{'on' if enabled else 'off'}.mp3")
        pygame.mixer.music.play()

    def run(self):
        while True:
            if self.data_queue.empty() or not self.key_listener.recognition:
                self.data_queue.queue.clear()
                sleep(0.01)
            else:
                start = time.time()
                audio_data = b''.join(self.data_queue.queue)
                self.data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                upper_case = False
                if self.key_listener.shift_pressed or time.time() - self.key_listener.last_shift_press < len(audio_np) / 16000:
                    upper_case = True
                original_text = self.audio_model.transcribe(audio_np, language=self.language).strip()
                text = original_text

                text = text.rstrip('.').strip()
                text = (text[:1].upper() if upper_case else text[:1].lower()) + text[1:]

                # Replace punctuations
                for punctuation, parts in self.args.punctuations.items():
                    for part in parts:
                        if part in text.lower():
                            text = re.sub(" *" + part + "\S*", punctuation, text, flags=re.IGNORECASE)

                text = re.sub(r'\s([:.])', r'\1', text)

                # Check if text matches one of the values and set params.language
                language_changed = False
                if " " not in text:
                    for l, values in self.args.languages.items():
                        if any(re.search(value, text, re.IGNORECASE) for value in values):
                            self.language = l
                            language_changed = True

                if any([v in text.lower() for v in self.args.hallucination_parts]):
                    text = ""
                if any([v == text.lower() for v in self.args.hallucinations]):
                    text = ""

                print(f"[{len(audio_np) / 16000:3.1f}s, {time.time() - start:3.1f}s] <{original_text}> -> <{text}>")
                if text:
                    pygame.mixer.music.load(f"{self.current_file_directory}/sounds/{self.language + '.mp3' if language_changed else 'done.wav'}")
                    pygame.mixer.music.play()
                    pyperclip.copy(text)

if __name__ == "__main__":
    transcriber = Transcriber()
    transcriber.run()
