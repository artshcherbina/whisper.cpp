# Whisper Speech Recognition

This Python application uses OpenAI's Whisper ASR model to transcribe spoken language into text.

The application also includes features for punctuation replacement and hallucination filtering.
It provides real-time transcription and supports multiple languages including English, Ukrainian, Korean and Russian. The transcribed text is copied to the clipboard and a sound is played to indicate the completion of transcription.

Press `Shift` key during recording to make the text start from an uppercase letter, otherwise, it will start from lowercase.

To switch languages, simply say the name of the language (English, Ukrainian, Korean or Russian) as a single word during recording. The application will detect the language name and switch to that language for transcription.

Configuration is made easy through a JSON file.
Available options:

- `energy_threshold`: This option sets the energy level threshold for the microphone to detect speech. It's an integer value and the default is 100.

- `record_timeout`: This option sets the real-time duration of the recording in seconds. It's a floating-point value and the default is 30.

- `pause_timeout`: This option sets the amount of empty space (in seconds) between recordings before considering it a new line in the transcription. It's a floating-point value and the default is 0.5.

- `no_gpu`: This is a boolean option that, when set to true, disables GPU usage. The default is false, meaning that GPU usage is enabled by default.

- `default_microphone`: This option sets the default microphone name for SpeechRecognition. If you're on a Linux platform, you can run the script with 'list' to view available microphones. The default value is 'pulse'.

These options should be set in the [`config.json`](./config.json) file in the following format:

```json
{
    "model": "medium",
    "energy_threshold": 100,
    "record_timeout": 30,
    "pause_timeout": 0.5,
    "no_gpu": false,
    "default_microphone": "pulse"
}
```
