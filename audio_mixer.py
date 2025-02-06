import librosa
import sounddevice as sd
import numpy as np
import time


class AudioMixer:

    playback_stream = None
    current_audio = None
    sample_rate = None
    volume = 1.0
    is_playing = False
    current_position = 0

    @staticmethod

    def play_audio(file_path, fade_in_duration=2):
        """Play audio with fade-in effect"""
        try:
            y, sr = librosa.load(file_path, sr=None)
            y = y.astype(np.float64)  # need to cconvert to float 64 cause sounddevice expects 64 and librosa is 32
            AudioMixer.current_audio = y
            AudioMixer.sample_rate = sr
            AudioMixer.playback_stream = sd.OutputStream(samplerate=sr, channels=1)

            AudioMixer.playback_stream.start()
            AudioMixer.is_playing = True
            AudioMixer.current_position = 0  # Reset position

            AudioMixer._fade_in(y, fade_in_duration)

            AudioMixer.playback_stream.write(y[int(sr * fade_in_duration):])

        except Exception as e:
            print(f"Error playing audio: {e}")

    @staticmethod
    def stop_audio(fade_out_duration=2):
        """Stop audio fade-out effect"""
        try:
            if AudioMixer.playback_stream is not None and AudioMixer.is_playing:
                AudioMixer._fade_out(AudioMixer.current_audio, fade_out_duration)
                AudioMixer.playback_stream.stop()
                AudioMixer.playback_stream.close()
                AudioMixer.playback_stream = None
                AudioMixer.current_audio = None
                AudioMixer.is_playing = False
                print("Audio playback stopped.")
            else:
                print("No audio is currently playing.")
        except Exception as e:
            print(f"Error stopping audio: {e}")

    @staticmethod
    def set_volume(level):
        try:
            volume_level = float(level)  # Convert to float
            AudioMixer.volume = np.clip(volume_level, 0.0, 1.0)
            if AudioMixer.playback_stream is not None and AudioMixer.is_playing:
                AudioMixer.playback_stream.write(AudioMixer.current_audio * AudioMixer.volume)
        except ValueError:
            print(f"Invalid volume level: {level}")

    @staticmethod
    def skip_to_position(position):
        """NEEDS further improving"""
        if AudioMixer.playback_stream is not None and AudioMixer.is_playing:
            try:
                position_samples = int(position * AudioMixer.sample_rate)
                AudioMixer.current_position = position_samples
                AudioMixer.playback_stream.write(AudioMixer.current_audio[position_samples:])
            except Exception as e:
                print(f"Error skipping to position: {e}")

    @staticmethod
    def _fade_in(audio_data, duration):
        num_samples = int(AudioMixer.sample_rate * duration)
        for i in range(num_samples):
            fade_level = i / num_samples
            sample_chunk = audio_data[i:i+1] * fade_level * AudioMixer.volume
            AudioMixer.playback_stream.write(sample_chunk)

    @staticmethod
    def _fade_out(audio_data, duration):
        num_samples = int(AudioMixer.sample_rate * duration)
        total_samples = len(audio_data)
        for i in range(num_samples):
            fade_level = 1.0 - (i / num_samples)
            sample_chunk = audio_data[total_samples - num_samples + i:total_samples - num_samples + i + 1] * fade_level * AudioMixer.volume
            AudioMixer.playback_stream.write(sample_chunk)