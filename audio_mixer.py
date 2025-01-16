# audio_mixer.py

import librosa
import sounddevice as sd

class AudioMixer:
    """Class for handling audio mixing and playback."""

    # A variable to manage the audio stream
    playback_stream = None

    @staticmethod
    def play_audio(file_path):
        try:
            y, sr = librosa.load(file_path)
            AudioMixer.playback_stream = sd.OutputStream(samplerate=sr, channels=len(y.shape) if y.ndim > 1 else 1)
            AudioMixer.playback_stream.start()
            AudioMixer.playback_stream.write(y)
        except Exception as e:
            print(f"Error playing audio: {e}")

    @staticmethod
    def stop_audio():
        try:
            if AudioMixer.playback_stream is not None:
                AudioMixer.playback_stream.stop()
                AudioMixer.playback_stream.close()
                AudioMixer.playback_stream = None
                print("Audio playback stopped.")
            else:
                print("No audio is currently playing.")
        except Exception as e:
            print(f"Error stopping audio: {e}")
