import os
import json
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from drop_detector import detect_drops  # Import the specialized drop detector

# run in command line for now  python section_analyzer.py ../riot --output results.json --visualize

def detect_sections(y, sr, drop_times=None):
    """
    EDM section detection algorithm that works across various EDM subgenres

    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    drop_times : list or None
        Optional list of pre-detected drop timestamps

    Returns:
    --------
    sections : list
        List of dictionaries containing section information
    """
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Spectral contrast for detecting low, mid, high frequencies
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bass_band = spec_contrast[0]  # First band is lowest frequencies

    # detecting rhythm changes
    percussive_rms = librosa.feature.rms(y=y_percussive, frame_length=n_fft, hop_length=hop_length)[0]

    # spectral flux
    spectral_flux = np.diff(np.abs(D), axis=1) ** 2
    spectral_flux = np.sum(spectral_flux, axis=0)
    spectral_flux = np.concatenate(([0], spectral_flux))

    novelty = (0.3 * librosa.util.normalize(rms) +
               0.3 * librosa.util.normalize(spectral_flux) +
               0.2 * librosa.util.normalize(bass_band) +
               0.2 * librosa.util.normalize(percussive_rms))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    smooth_window = int(sr // hop_length * (120 / max(tempo, 1)))

    novelty_smooth = np.convolve(novelty, np.ones(smooth_window) / smooth_window, mode='same')

    # Find peaks more reliably
    delta_val = 0.05  # Start with a modest threshold
    wait_val = int(8 * sr / hop_length)  # wait a few seconds between sections

    for attempt in range(3):
        peaks = librosa.util.peak_pick(
            novelty_smooth,
            pre_max=sr // hop_length,
            post_max=sr // hop_length,
            pre_avg=2 * sr // hop_length,
            post_avg=2 * sr // hop_length,
            delta=delta_val,
            wait=wait_val
        )

        if len(peaks) >= 2 and len(peaks) <= 10:
            break

        if len(peaks) < 2:
            delta_val /= 2
            wait_val = int(wait_val * 0.7)
        elif len(peaks) > 10:
            delta_val *= 1.5

    # Convert peaks to time
    boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    if len(boundary_times) == 0 or boundary_times[0] > 5:
        boundary_times = np.concatenate(([0], boundary_times))

    track_length = librosa.get_duration(y=y, sr=sr)
    if len(boundary_times) == 0 or boundary_times[-1] < track_length - 5:
        boundary_times = np.concatenate((boundary_times, [track_length]))

    # Merge boundaries that are too close together
    if len(boundary_times) > 2:
        # Minimum section length in seconds (typically around 8 bars in EDM)
        min_section_length = 60 * 8 / tempo  # 8 bars at detected tempo
        min_section_length = max(10, min_section_length)  # At least 10 seconds

        i = 0
        while i < len(boundary_times) - 1:
            if boundary_times[i + 1] - boundary_times[i] < min_section_length:
                boundary_times = np.delete(boundary_times, i + 1)
            else:
                i += 1

    # Calculate bass enhancement and rhythmic intensity
    bass_enhancement = np.zeros_like(rms)
    for i in range(1, len(bass_band)):
        if i < len(bass_band):
            # Calculate how much the bass increases at each point
            bass_enhancement[i] = max(0, bass_band[i] - bass_band[i - 1])

    # Detect bass drops, places where bass suddenly becomes strong
    bass_peaks = librosa.util.peak_pick(
        librosa.util.normalize(bass_band),
        pre_max=sr // hop_length,
        post_max=sr // hop_length,
        pre_avg=sr // hop_length * 2,
        post_avg=sr // hop_length * 2,
        delta=0.4,
        wait=sr // hop_length * 4
    )
    bass_peak_times = librosa.frames_to_time(bass_peaks, sr=sr, hop_length=hop_length)

    # Classify each section
    sections = []

    # Calculate overall energy statistics for relative comparisons
    rms_percentiles = np.percentile(rms, [20, 40, 60, 80])

    for i in range(len(boundary_times) - 1):
        start_time = boundary_times[i]
        end_time = boundary_times[i + 1]

        # Get segment indices (accounting for hop_length)
        start_idx = librosa.time_to_frames(start_time, sr=sr, hop_length=hop_length)
        end_idx = librosa.time_to_frames(end_time, sr=sr, hop_length=hop_length)

        # Safety check for valid indices
        if start_idx >= len(rms) or start_idx >= end_idx:
            continue

        end_idx = min(end_idx, len(rms) - 1)  # Ensure we don't go past the end

        # Extract segment features
        segment_rms = np.mean(rms[start_idx:end_idx])
        segment_bass = np.mean(bass_band[start_idx:end_idx]) if start_idx < len(bass_band) else 0
        segment_perc = np.mean(percussive_rms[start_idx:end_idx]) if start_idx < len(percussive_rms) else 0

        # For buildups, check if energy is increasing
        energy_slope = 0
        if end_idx - start_idx > 20:
            x = np.arange(end_idx - start_idx)
            energy_slope, _ = np.polyfit(x, rms[start_idx:end_idx], 1)

        # Calculate relative position in track
        position = i / (len(boundary_times) - 1)
        duration = end_time - start_time

        # Classify based on multiple features
        # Use relative energy levels compared to the track as a whole
        energy_level = 0
        for j, p in enumerate(rms_percentiles):
            if segment_rms > p:
                energy_level = j + 1

        # First check if this section contains any pre-detected drops
        is_drop = False
        if drop_times:
            for drop_time in drop_times:
                if start_time <= drop_time <= end_time:
                    section_type = "drop"
                    is_drop = True
                    break

        if not is_drop:
            # Classification logic that works for various EDM subgenres
            if position < 0.15:
                # First section is typically an intro
                section_type = "intro"
            elif position > 0.85:
                # Last section is typically an outro
                section_type = "outro"
            elif (energy_level >= 3 and segment_bass > np.percentile(bass_band, 65)) or \
                    (segment_perc > np.percentile(percussive_rms, 75) and segment_bass > np.mean(bass_band)) or \
                    any(abs(bp - start_time) < 5 for bp in bass_peak_times):  # Section starts near a bass peak
                # High energy with strong bass is typically a drop
                section_type = "drop"
            elif energy_slope > 0.0005 and duration > 12:
                # Increasing energy over time is typically a buildup
                section_type = "buildup"
            elif energy_level <= 1 and position > 0.15:
                # Low energy section after intro is typically a breakdown
                section_type = "breakdown"
            else:
                # Default to verse for other sections
                section_type = "verse"

        # Create section info
        section_info = {
            'type': section_type,
            'start': float(start_time),
            'end': float(end_time),
            'duration': float(duration),
            'energy': float(segment_rms),
            'energy_level': int(energy_level),
            'bass_energy': float(segment_bass),
            'percussive_energy': float(segment_perc)
        }

        sections.append(section_info)

    return sections


def visualize_sections(audio_path, sections, save_path=None):
    """Visualize detected sections and save to file if requested"""
    y, sr = librosa.load(audio_path, sr=44100)

    plt.figure(figsize=(14, 10))

    # Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title(f"Waveform with Sections: {os.path.basename(audio_path)}")

    section_colors = {
        "intro": "#3498db",
        "buildup": "#f39c12",
        "drop": "#e74c3c",
        "breakdown": "#9b59b6",
        "verse": "#2ecc71",
        "outro": "#34495e"
    }

    y_min, y_max = plt.ylim()

    for section in sections:
        color = section_colors.get(section['type'], "#7f8c8d")
        plt.axvspan(section['start'], section['end'], alpha=0.3, color=color)
        middle_x = (section['start'] + section['end']) / 2
        plt.text(middle_x, y_max * 0.8, section['type'], horizontalalignment='center',
                 fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7))

    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram with Sections")

    # Add colored sections to spectrogram
    for section in sections:
        color = section_colors.get(section['type'], "#7f8c8d")
        plt.axvspan(section['start'], section['end'], alpha=0.3, color=color)

    # Plot energy and bass content
    plt.subplot(3, 1, 3)

    # Get energy curve
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Get bass content
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    bass_band = spec_contrast[0]  # First band is lowest frequencies

    plt.plot(times, librosa.util.normalize(rms), label='Energy', color='blue', linewidth=2)
    plt.plot(times[:len(bass_band)], librosa.util.normalize(bass_band), label='Bass', color='red', linewidth=1)

    plt.title("Energy and Bass Content")
    plt.legend()

    for section in sections:
        color = section_colors.get(section['type'], "#7f8c8d")
        plt.axvspan(section['start'], section['end'], alpha=0.3, color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


@dataclass
class TrackAnalysis:
    file_name: str
    sections: List[Dict[str, Any]]
    tempo: Optional[float] = None
    key: Optional[str] = None
    drop_times: Optional[List[float]] = None


def analyze_track(audio_path, detect_drops_first=True):
    """Analyze a single track and return the analysis data"""
    print(f"Analyzing {os.path.basename(audio_path)}...")

    y, sr = librosa.load(audio_path, sr=44100)

    # Get tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    drop_times = None
    if detect_drops_first:
        try:
            drops = detect_drops(audio_path, plot=False)
            if drops:
                drop_times = [drop['time'] for drop in drops]
                print(f"  Pre-detected {len(drop_times)} drops at: {', '.join([f'{t:.1f}s' for t in drop_times])}")
        except Exception as e:
            print(f"  Error in drop detection: {e}")
            drop_times = None

    sections = detect_sections(y, sr, drop_times)

    return TrackAnalysis(
        file_name=os.path.basename(audio_path),
        sections=sections,
        tempo=float(tempo),
        drop_times=drop_times
    )


def analyze_folder(folder_path, output_file=None, visualize=False, visualization_folder=None, detect_drops_first=True):
    """Analyze all audio files in a folder and save results to a file"""
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.mp3', '.wav'))]

    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return

    print(f"Found {len(audio_files)} audio files in {folder_path}")

    if visualize and visualization_folder:
        os.makedirs(visualization_folder, exist_ok=True)

    # Analyzing each track
    results = []
    for audio_file in tqdm(audio_files, desc="Analyzing tracks"):
        analysis = analyze_track(audio_file, detect_drops_first)
        results.append(analysis)

        # Generating visualization
        if visualize:
            if visualization_folder:
                base_name = os.path.splitext(analysis.file_name)[0]
                viz_path = os.path.join(visualization_folder, f"{base_name}_sections.png")
            else:
                viz_path = None

            visualize_sections(audio_file, analysis.sections, viz_path)

    # Saving results to file
    if output_file:
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=4)
        elif output_file.endswith('.csv'):
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'Tempo', 'Section Count', 'Section Types', 'Drop Times'])

                for result in results:
                    section_types = ', '.join(set(s['type'] for s in result.sections))
                    drop_times_str = ', '.join([f"{t:.1f}" for t in result.drop_times]) if result.drop_times else ""
                    writer.writerow([
                        result.file_name,
                        result.tempo,
                        len(result.sections),
                        section_types,
                        drop_times_str
                    ])

        print(f"Analysis results saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze EDM tracks to detect song sections")
    parser.add_argument('folder', help='Folder containing audio files to analyze')
    parser.add_argument('--output', '-o', help='Output file to save analysis results (.json or .csv)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations of detected sections')
    parser.add_argument('--viz-folder', help='Folder to save visualizations')
    parser.add_argument('--no-drop-detection', action='store_true', help='Disable specialized drop detection')

    args = parser.parse_args()

    results = analyze_folder(
        args.folder,
        args.output,
        args.visualize,
        args.viz_folder,
        not args.no_drop_detection
    )

    print(f"Analysis complete. Processed {len(results)} tracks.")