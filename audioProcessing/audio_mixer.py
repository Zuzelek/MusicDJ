import pygame
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import time


class AudioMixer:
    """Class for audio mixing and playback operations."""

    _playing = False
    _current_stream = None
    _volume = 1.0
    _stop_requested = False

    @staticmethod
    def set_volume(vol):
        """Set the global volume level."""
        AudioMixer._volume = float(vol)

    @staticmethod
    def play_audio(file_path, fade_in_duration=0, fade_out_duration=2):
        """Play audio with optional fade in/out."""
        AudioMixer.stop_audio()

        AudioMixer._stop_requested = False

        try:
            data, samplerate = sf.read(file_path)

            if fade_in_duration > 0:
                fade_in_samples = int(fade_in_duration * samplerate)
                if fade_in_samples < len(data):
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                    if len(data.shape) > 1:  # Stereo
                        data[:fade_in_samples, 0] *= fade_in_curve
                        data[:fade_in_samples, 1] *= fade_in_curve
                    else:  # Mono
                        data[:fade_in_samples] *= fade_in_curve

            AudioMixer._playing = True

            # Start playback
            AudioMixer._current_stream = sd.OutputStream(
                samplerate=samplerate,
                channels=data.shape[1] if len(data.shape) > 1 else 1,
                callback=lambda outdata, frames, time, status:
                AudioMixer._audio_callback(outdata, frames, time, status, data, samplerate, fade_out_duration)
            )
            AudioMixer._current_stream.start()

        except Exception as e:
            print(f"Error playing audio: {e}")

    @staticmethod
    def _audio_callback(outdata, frames, time, status, audio_data, samplerate, fade_out_duration):
        """Callback for audio playback."""
        if status:
            print(status)

        if not AudioMixer._playing or AudioMixer._stop_requested:
            # Apply fade out
            if fade_out_duration > 0:
                fade_out_samples = int(fade_out_duration * samplerate)
                if fade_out_samples < len(audio_data):
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                    if len(audio_data.shape) > 1:  # Stereo
                        audio_data[-fade_out_samples:, 0] *= fade_out_curve
                        audio_data[-fade_out_samples:, 1] *= fade_out_curve
                    else:  # Mono
                        audio_data[-fade_out_samples:] *= fade_out_curve

            # Close the stream
            raise sd.CallbackStop()

        # Fill the output buffer
        if len(outdata) <= len(audio_data):
            outdata[:] = audio_data[:len(outdata)] * AudioMixer._volume
            audio_data = audio_data[len(outdata):]
        else:
            outdata[:len(audio_data)] = audio_data * AudioMixer._volume
            outdata[len(audio_data):] = 0
            AudioMixer._playing = False

    @staticmethod
    def stop_audio(fade_out_duration=2):
        """Stop audio playback with optional fade out."""
        if AudioMixer._playing and AudioMixer._current_stream:
            AudioMixer._stop_requested = True

            if fade_out_duration > 0:
                time.sleep(fade_out_duration)

            AudioMixer._current_stream.stop()
            AudioMixer._current_stream.close()
            AudioMixer._current_stream = None
            AudioMixer._playing = False

    @staticmethod
    def skip_to_position(position):
        """Skip to a specific position in the track."""
        # will be implemented later on.
        pass

    @staticmethod
    def play_section(file_path, start_time, end_time, fade_in=0.5, fade_out=0.5):
        """
        Play a specific section of an audio file

        Parameters:
        -----------
        file_path : str
            Path to the audio file
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        fade_in : float
            Fade-in duration in seconds
        fade_out : float
            Fade-out duration in seconds
        """
        AudioMixer.stop_audio()

        try:
            data, samplerate = sf.read(file_path)

            start_sample = int(start_time * samplerate)
            end_sample = int(end_time * samplerate)

            start_sample = max(0, min(start_sample, len(data) - 1))
            end_sample = max(0, min(end_sample, len(data)))

            section = data[start_sample:end_sample]

            if fade_in > 0:
                fade_in_samples = int(fade_in * samplerate)
                if fade_in_samples < len(section):
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                    if len(section.shape) > 1:  # Stereo
                        section[:fade_in_samples, 0] *= fade_in_curve
                        section[:fade_in_samples, 1] *= fade_in_curve
                    else:  # Mono
                        section[:fade_in_samples] *= fade_in_curve

            if fade_out > 0:
                fade_out_samples = int(fade_out * samplerate)
                if fade_out_samples < len(section):
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                    if len(section.shape) > 1:  # Stereo
                        section[-fade_out_samples:, 0] *= fade_out_curve
                        section[-fade_out_samples:, 1] *= fade_out_curve
                    else:  # Mono
                        section[-fade_out_samples:] *= fade_out_curve

            sd.play(section * AudioMixer._volume, samplerate)

        except Exception as e:
            print(f"Error playing section: {e}")

    @staticmethod
    def crossfade(file_path1, file_path2, crossfade_duration=5, play=True):
        """
        Create a crossfade between two audio files

        Parameters:
        -----------
        file_path1 : str
            Path to the first audio file
        file_path2 : str
            Path to the second audio file
        crossfade_duration : float
            Duration of crossfade in seconds
        play : bool
            Whether to play the result immediately

        Returns:
        --------
        mixed_audio : numpy.ndarray
            The resulting audio with crossfade
        samplerate : int
            Sample rate of the audio
        """
        try:
            # Load audio files
            data1, sr1 = sf.read(file_path1)
            data2, sr2 = sf.read(file_path2)

            if sr1 != sr2:
                print("Warning: Sample rates don't match. Resampling may be needed for best results.")

            # Use the first sample rate for output
            samplerate = sr1

            xfade_samples = int(crossfade_duration * samplerate)

            if len(data1) < xfade_samples:
                print(f"Warning: First track is shorter than crossfade duration")
                xfade_samples = len(data1)

            # Create crossfade curves
            fade_out = np.linspace(1, 0, xfade_samples)
            fade_in = np.linspace(0, 1, xfade_samples)

            # Apply crossfade
            end_of_first = data1[-xfade_samples:]
            start_of_second = data2[:xfade_samples]

            if len(data1.shape) > 1 and len(data2.shape) > 1:  # Both stereo
                for i in range(min(data1.shape[1], data2.shape[1])):
                    end_of_first[:, i] *= fade_out
                    start_of_second[:, i] *= fade_in

                # Mix the crossfade region
                crossfade_mix = end_of_first + start_of_second

                # Combine everything
                mixed_audio = np.vstack((data1[:-xfade_samples], crossfade_mix, data2[xfade_samples:]))

            else:
                if len(data1.shape) > 1:
                    end_of_first = np.mean(end_of_first, axis=1)
                if len(data2.shape) > 1:
                    start_of_second = np.mean(start_of_second, axis=1)

                end_of_first *= fade_out
                start_of_second *= fade_in

                crossfade_mix = end_of_first + start_of_second

                mixed_audio = np.concatenate((data1[:-xfade_samples], crossfade_mix, data2[xfade_samples:]))

            # Play if requested
            if play:
                sd.play(mixed_audio * AudioMixer._volume, samplerate)

            return mixed_audio, samplerate

        except Exception as e:
            print(f"Error creating crossfade: {e}")
            return None, None

    @staticmethod
    def beatmatch(file_path1, file_path2, target_bpm=None, play=True):

        try:
            # need to implement later
            # 1. Detect BPM for both tracks
            # 2. Time-stretch the second track to match the BPM of the first (or target BPM)
            # 3. Align beats for a smooth transition
            # 4. Apply crossfade

            # For now, just have a simple crossfade
            return AudioMixer.crossfade(file_path1, file_path2, crossfade_duration=8, play=play)

        except Exception as e:
            print(f"Error beatmatching tracks: {e}")
            return None, None