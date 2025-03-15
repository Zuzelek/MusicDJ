import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import soundfile as sf


class FrequencyProcessor:
    """
    My functions for manipulating frequencies during DJ mixing.
    Includes filters for EQ and transition effects.
    """

    def __init__(self):
        pass

    def butter_highpass(self, cutoff, fs, order=5):
        # Design highpass filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        # Apply highpass filter
        b, a = self.butter_highpass(cutoff, fs, order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass(self, cutoff, fs, order=5):
        # Design lowpass filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        # Apply lowpass filter
        b, a = self.butter_lowpass(cutoff, fs, order)
        y = lfilter(b, a, data)
        return y

    def butter_high_shelf(self, cutoff, fs, gain=1.0, order=5):
        # High shelf filter for boosting highs
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        b = b * gain  # Apply gain
        return b, a

    def butter_high_shelf_filter(self, data, cutoff, fs, gain, order=5):
        b, a = self.butter_high_shelf(cutoff, fs, gain, order)
        y = lfilter(b, a, data)
        return y

    def butter_low_shelf(self, cutoff, fs, gain=1.0, order=5):
        # Low shelf filter for boosting bass
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        b = b * gain  # Apply gain
        return b, a

    def butter_low_shelf_filter(self, data, cutoff, fs, gain, order=5):
        b, a = self.butter_low_shelf(cutoff, fs, gain, order)
        y = lfilter(b, a, data)
        return y

    def reduce_low_frequencies(self, y, sr, cutoff=100, plot=False):
        # Kill the bass - useful for transitions
        y_filtered = self.butter_highpass_filter(y, cutoff, sr)

        if plot:
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y_filtered, sr=sr)
            plt.title(f"After High-Pass Filter (Reduced Low Frequencies below {cutoff}Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()

        return y_filtered

    def reduce_high_frequencies(self, y, sr, cutoff=1000, plot=False):
        # Cut the highs - smoother transitions
        y_filtered = self.butter_lowpass_filter(y, cutoff, sr)

        if plot:
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y_filtered, sr=sr)
            plt.title(f"After Low-Pass Filter (Reduced High Frequencies above {cutoff}Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()

        return y_filtered

    def increase_high_frequencies(self, y, sr, cutoff=1000, gain=2.0, plot=False):
        # Boost highs - adds clarity/brightness
        y_filtered = self.butter_high_shelf_filter(y, cutoff, sr, gain)

        if plot:
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y_filtered, sr=sr)
            plt.title(f"After High-Shelf Filter (Increased High Frequencies above {cutoff}Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()

        return y_filtered

    def increase_low_frequencies(self, y, sr, cutoff=100, gain=2.0, plot=False):
        # Boost bass - adds power/impact
        y_filtered = self.butter_low_shelf_filter(y, cutoff, sr, gain)

        if plot:
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y_filtered, sr=sr)
            plt.title(f"After Low-Shelf Filter (Increased Low Frequencies below {cutoff}Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()

        return y_filtered

    def save_processed_audio(self, y, sr, output_path):
        # Save the processed audio
        sf.write(output_path, y, sr)