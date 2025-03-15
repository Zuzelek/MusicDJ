import os
import numpy as np
import librosa
import soundfile as sf
import csv
import json
from pathlib import Path


def load_audio(file_path, sr=44100):
    # Safe audio loading with error handling
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None


def match_target_tempo(y, sr, source_bpm, target_bpm):
    # Match one track's tempo to another
    ratio = target_bpm / source_bpm
    return librosa.effects.time_stretch(y, rate=ratio)


def crossfade(y1, y2, sr, fade_duration=8.0):
    # Smooth transition between tracks
    fade_samples = int(fade_duration * sr)

    # Make sure we have enough audio to fade
    if len(y1) < fade_samples or len(y2) < fade_samples:
        raise ValueError("Audio segments too short for this crossfade length")

    # Create fades
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    # Apply fades
    y1_end = y1[-fade_samples:]
    y1_end_faded = y1_end * fade_out

    y2_start = y2[:fade_samples]
    y2_start_faded = y2_start * fade_in

    # Combine everything
    crossfaded = y1_end_faded + y2_start_faded
    result = np.concatenate([y1[:-fade_samples], crossfaded, y2[fade_samples:]])

    return result


def export_to_csv(data, file_path):
    # Quick export to CSV for later analysis
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "BPM", "Key", "Energy"])

        for item in data:
            writer.writerow([
                item.get('file', ''),
                item.get('bpm', ''),
                item.get('key', ''),
                item.get('energy', '')
            ])


def export_to_json(data, file_path):
    # Export to JSON for better data structure preservation
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def create_mix_from_transitions(transitions, output_path):
    # Create a full mix from my transition points
    mix = np.array([])

    for i, transition in enumerate(transitions):
        # Load current track
        y_current, sr = load_audio(transition['track_path'])

        if i == 0:
            # First track - include everything up to exit
            exit_sample = int(transition['exit_point'] * sr)
            mix = np.concatenate([mix, y_current[:exit_sample]])
        else:
            # Subsequent tracks - create transition
            prev_track = transitions[i - 1]['track_path']
            y_prev, sr_prev = load_audio(prev_track)

            # Get exit and entry points
            exit_sample = int(transitions[i - 1]['exit_point'] * sr_prev)
            entry_sample = int(transition['entry_point'] * sr)

            # Crossfade duration
            crossfade_duration = transition.get('crossfade_duration', 8.0)

            # Extract segments for crossfade
            prev_segment = y_prev[exit_sample - int(crossfade_duration * sr_prev):]
            current_segment = y_current[:entry_sample + int(crossfade_duration * sr)]

            # Create smooth transition
            fade_result = crossfade(prev_segment, current_segment, sr, crossfade_duration)

            # Add to mix
            mix = np.concatenate([mix[:-int(crossfade_duration * sr)], fade_result])

    # Save final result
    sf.write(output_path, mix, sr)

    return mix