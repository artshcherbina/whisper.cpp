#! python3.7

import argparse
import os
import re
import time
import pyperclip

import numpy as np
import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

from scipy.io import wavfile
from whisper_cpp import WhisperCpp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=100,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=30,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=0.5,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--no-gpu", help="Disable GPU usage", action="store_true")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
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
    audio_model = WhisperCpp(model="/home/artem/git/whisper.cpp/models/ggml-large-v3-q5_0.bin", use_gpu=not args.no_gpu)

    record_timeout = args.record_timeout

    transcription = ['']

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

    while True:
        try:
            # Pull raw recorded audio from the queue.
            if data_queue.empty():
                # Infinite loops are bad for processors, must sleep.
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

                # Read the transcription.
                original_text = audio_model.transcribe(audio_np, language=language).strip()
                text = original_text

                text = text.rstrip('.').strip()
                text = text[:1].upper() + text[1:]

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
                    "ru": ["Russian", "Rusk", "Русский", "русский"],
                    "en": ["English", "Англий", "англий"]
                }
                language_changed = False
                if " " not in text:
                    for l, values in languages.items():
                        if any(re.search(value, text, re.IGNORECASE) for value in values):
                            language = l
                            language_changed = True

                pyperclip.copy(text)
                print(f"[{len(audio_np) / 16000:3.1f}s, {time.time() - start:3.1f}s] <{original_text}> -> <{text}>") 
                os.system(f"play /home/artem/Downloads/{language if language_changed else 'beep'}.mp3 &> /dev/null &")
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
