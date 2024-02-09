import cv2
from pathlib import Path

from matplotlib import pyplot as plt

from RealtimeTTS import BaseEngine
import pyaudio
import piper


class PiperEngine(BaseEngine):
    def __init__(self, models_folder):
        self.engine_name = "piper"
        self.path_voice = Path(models_folder) / 'en_GB-alba-medium.onnx'
        self.engine = piper.PiperVoice.load(self.path_voice, config_path=None, use_cuda=False)

        # Indicates if the engine can handle generators.
        self.can_consume_generators = False

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for System Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """
        return pyaudio.paInt16, 1, 22050

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        audio_stream = self.engine.synthesize_stream_raw(text)
        for audio_bytes in audio_stream:
            self.queue.put(audio_bytes)
        return True


def display(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title('my picture')
    plt.show()
