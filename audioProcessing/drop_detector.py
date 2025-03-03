import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os


def detect_drops(audio_path, plot=True, save_plot=None):

    print(f"Analyzing drops in: {os.path.basename(audio_path)}")

    # Load audio with higher sample rate for better frequency resolution
    y, sr = librosa.load(audio_path, sr=44100)

    # Extract primary features
    # 1. Energy (RMS)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # 2. Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    perc_rms = librosa.feature.rms(y=y_percussive, hop_length=hop_length)[0]

    # 3. Spectral contrast - for bass detection
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    bass_band = spec_contrast[0]  # First band is lowest frequencies

    # 4. Spectral bandwidths for different frequency ranges
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]

    # 5. Sub-bass energy (20-60Hz)
    n_fft = 4096  # Larger FFT for better low frequency resolution
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Find bins corresponding to sub-bass range
    sub_bass_bins = np.where((freqs >= 20) & (freqs <= 60))[0]
    sub_bass_energy = np.mean(np.abs(stft[sub_bass_bins, :]), axis=0)

    # 6. Onset strength and backbeat detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    # Compute multiple drop indicators
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Indicator 1: Bass enhancement (where bass increases significantly)
    bass_diff = np.diff(librosa.util.normalize(bass_band), prepend=0)
    bass_increase = np.zeros_like(bass_diff)
    bass_increase[bass_diff > 0] = bass_diff[bass_diff > 0]

    # Indicator 2: Sub-bass to mid-high ratio (drops often have strong sub-bass)
    sub_bass_ratio = sub_bass_energy / (np.mean(np.abs(stft), axis=0) + 1e-8)

    # Indicator 3: Energy jumps
    energy_diff = np.diff(librosa.util.normalize(rms), prepend=0)
    energy_jumps = np.zeros_like(energy_diff)
    energy_jumps[energy_diff > 0] = energy_diff[energy_diff > 0]

    # Indicator 4: Percussion intensity
    perc_intensity = librosa.util.normalize(perc_rms)

    # Indicator 5: Structural changes - look for drops after quiet sections
    local_mean = np.convolve(rms, np.ones(sr // hop_length) / (sr // hop_length), mode='same')
    structural_jumps = rms / (local_mean + 1e-8)

    # Combined drop score - weighted combination of indicators
    drop_score = (0.2 * bass_increase +
                  0.25 * librosa.util.normalize(sub_bass_ratio) +
                  0.2 * energy_jumps +
                  0.15 * perc_intensity +
                  0.2 * librosa.util.normalize(structural_jumps))

    # Smooth the score
    drop_score_smooth = np.convolve(drop_score, np.ones(sr // hop_length // 2) / (sr // hop_length // 2), mode='same')

    # Find peaks in the drop score
    # Use beat-aware peak finding to find drops that align with musical structure
    peak_indices, _ = find_peaks(
        drop_score_smooth,
        height=0.5,  # Minimum height
        distance=sr // hop_length * 4  # Minimum distance between drops (4 seconds)
    )

    peak_times = times[peak_indices]
    peak_scores = drop_score_smooth[peak_indices]

    # Filter out peaks that are too close to the beginning or end
    min_time = 20  # Drops typically don't happen in first 20 seconds
    max_time = librosa.get_duration(y=y, sr=sr) - 10  # Or last 10 seconds

    drops = []
    for i, (time, score) in enumerate(zip(peak_times, peak_scores)):
        if min_time <= time <= max_time:
            drops.append({
                'time': float(time),
                'score': float(score),
                'confidence': 'high' if score > 0.7 else 'medium' if score > 0.5 else 'low'
            })

    # Sort drops by score
    drops = sorted(drops, key=lambda x: x['score'], reverse=True)

    # Get the best drops (up to 3)
    best_drops = drops[:min(3, len(drops))]

    # If we found no drops but should have, take the highest energy section
    if not best_drops and librosa.get_duration(y=y, sr=sr) > 90:  # Track longer than 90 seconds likely has a drop
        high_energy_times = times[np.argsort(rms)[-5:]]  # Get times of 5 highest energy points
        # Take the one furthest from start/end
        distances = np.minimum(high_energy_times, max_time - high_energy_times)
        best_time = high_energy_times[np.argmax(distances)]
        best_drops = [{
            'time': float(best_time),
            'score': 0.5,
            'confidence': 'inferred'
        }]

    # If visualization is requested
    if plot:
        plt.figure(figsize=(14, 10))

        # Plot waveform
        plt.subplot(5, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f"Waveform: {os.path.basename(audio_path)}")

        # Mark drops on waveform
        y_min, y_max = plt.ylim()
        for drop in best_drops:
            plt.axvline(x=drop['time'], color='r', linestyle='--', linewidth=2)
            plt.text(drop['time'], y_max * 0.8, f"DROP\n{drop['time']:.1f}s",
                     horizontalalignment='center', color='red',
                     bbox=dict(facecolor='white', alpha=0.8))

        # Plot spectrogram
        plt.subplot(5, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram with Drops")

        # Mark drops on spectrogram
        for drop in best_drops:
            plt.axvline(x=drop['time'], color='r', linestyle='--', linewidth=2)

        # Plot drop score
        plt.subplot(5, 1, 3)
        plt.plot(times, drop_score_smooth, label='Drop Score', color='purple', linewidth=2)
        plt.title("Drop Detection Score")
        plt.xlabel("Time (s)")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)

        for drop in best_drops:
            plt.plot(drop['time'], drop['score'], 'ro', markersize=10)
            plt.text(drop['time'], drop['score'], f"{drop['confidence']} ({drop['score']:.2f})",
                     horizontalalignment='center', verticalalignment='bottom')

        # Plot RMS energy
        plt.subplot(5, 1, 4)
        plt.plot(times, librosa.util.normalize(rms), label='Energy', color='blue', linewidth=2)
        plt.plot(times[:len(bass_band)], librosa.util.normalize(bass_band),
                 label='Bass', color='green', linewidth=1)
        plt.title("Energy and Bass Content")
        plt.legend()
        plt.grid(True, alpha=0.3)

        for drop in best_drops:
            plt.axvline(x=drop['time'], color='r', linestyle='--', linewidth=2)

        # Plot sub-bass content
        plt.subplot(5, 1, 5)
        plt.plot(times[:len(sub_bass_energy)], librosa.util.normalize(sub_bass_energy),
                 label='Sub-bass', color='orange', linewidth=2)
        plt.plot(times[:len(sub_bass_ratio)], librosa.util.normalize(sub_bass_ratio),
                 label='Sub-bass Ratio', color='brown', linewidth=1)
        plt.title("Sub-bass Content")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        for drop in best_drops:
            plt.axvline(x=drop['time'], color='r', linestyle='--', linewidth=2)

        plt.tight_layout()

        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_plot}")

        plt.show()