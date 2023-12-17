from pydub import AudioSegment
from pydub.generators import Sine

def create_suggested_melodies():
    # Define the note frequencies for the melodies (in Hz)
    note_frequencies = {
        'C4': 261.63, 'E4': 329.63, 'G4': 391.99, 'C5': 523.25,
        'B4': 493.88, 'D5': 587.33, 'G5': 783.99, 'D4': 293.66,
        'A4': 440.00, 'F4': 349.23, 'E5': 659.25
    }

    # Define the melodies
    melodies = [
        ['C4', 'E4', 'G4', 'C5'],             # C Major Arpeggio
        ['G4', 'B4', 'D5', 'G5'],             # G Major Arpeggio
        ['C4', 'D4', 'E4', 'G4', 'A4', 'C5'], # Pentatonic Ascend
        ['F4', 'B4', 'C5'],                   # Ascending Fourth Interval
        ['E5', 'C5'],                         # Descending Minor Third
        ['G4', 'A4', 'B4', 'G4', 'E4']        # Lyric Melody
    ]

    suggested_melody_paths = []

    for i, melody in enumerate(melodies):
        sound = AudioSegment.silent(duration=0)
        for note in melody:
            sound += Sine(note_frequencies[note]).to_audio_segment(duration=200)

        # Normalize and apply fade in/out
        sound = normalize(sound).fade_in(50).fade_out(50)

        # Export the sound to a file
        file_path = f"/mnt/data/suggested_melody_{i+1}.wav"
        sound.export(file_path, format="wav")
        suggested_melody_paths.append(file_path)

    return suggested_melody_paths

# Create and save the suggested melodies
suggested_melody_file_paths = create_suggested_melodies()
suggested_melody_file_paths

