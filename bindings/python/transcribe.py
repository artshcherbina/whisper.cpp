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

def main():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    with open(current_file_directory + '/config.json', 'r') as f:
        args = argparse.Namespace(**json.load(f))

    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer(args.energy_threshold, args.pause_timeout)
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    # args.model
    audio_model = WhisperCpp(model="/home/artem/git/whisper.cpp/models/ggml-large-v3-q5_0.bin", use_gpu=args.use_gpu)

    record_timeout = args.record_timeout

    # with source:
        # recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    language = "en"

    def recognition_callback(enabled):
        pygame.mixer.music.load(f"{current_file_directory}/sounds/{language}_{'on' if enabled else 'off'}.mp3")
        pygame.mixer.music.play()

    key_listener = KeyListener(recognition_callback)
    key_listener.start()

    while True:
        # Pull raw recorded audio from the queue.
        if data_queue.empty() or not key_listener.recognition:
            data_queue.queue.clear()
            sleep(0.01)
        else:
            start = time.time()
            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            upper_case = False
            if key_listener.shift_pressed or time.time() - key_listener.last_shift_press < len(audio_np) / 16000:
                upper_case = True
            # Read the transcription.
            original_text = audio_model.transcribe(audio_np, language=language).strip()
            text = original_text

            text = text.rstrip('.').strip()
            text = (text[:1].upper() if upper_case else text[:1].lower()) + text[1:]

            # Replace punctuations
            punctuations = {
                ":": ["colon", "двоето", "двойто"]
            }
            for punctuation, parts in punctuations.items():
                for part in parts:
                    if part in text.lower():
                        text = re.sub(" *" + part + "\S*", punctuation, text, flags=re.IGNORECASE)

            text = re.sub(r'\s([:.])', r'\1', text)

            # Check if text matches one of the values and set params.language
            languages = {
                "en": ["English", "Англий", "англий", "Англійсь", "영어"],
                "ru": ["Russian", "Rusk", "Русский", "русский", "російсь"],
                "uk": ["Ukrain", "украинск", "українс"],
                "ko": ["Korean", "корей"],
            }
            language_changed = False
            if " " not in text:
                for l, values in languages.items():
                    if any(re.search(value, text, re.IGNORECASE) for value in values):
                        language = l
                        language_changed = True

            if any([v in text.lower() for v in args.hallucination_parts]):
                text = ""
            if any([v == text.lower() for v in args.hallucinations]):
                text = ""

            print(f"[{len(audio_np) / 16000:3.1f}s, {time.time() - start:3.1f}s] <{original_text}> -> <{text}>")
            if text:
                pygame.mixer.music.load(f"{current_file_directory}/sounds/{language + '.mp3' if language_changed else 'done.wav'}")
                pygame.mixer.music.play()
                pyperclip.copy(text)

if __name__ == "__main__":
    main()
