import pyaudio
import wave
import time
from threading import Lock, Event


class Microphone:
    def __init__(self,
                 device_index=None):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 512
        self.device_index = device_index
        self._data = []
        self._lock = Lock()
        self._event = Event()
        self.max_buf_len = int(self.RATE * 60 / self.CHUNK)

        self.audio = None
        self.time_start = None
        self.stream = None

        # self.start()

    def start(self):
        self.audio = pyaudio.PyAudio()
        if self.device_index is None:
            print("-----------------------Select a device-----------------------")
            info = self.audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                    if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                        print("Input Device id ", i, " - ", self.audio.get_device_info_by_host_api_device_index(0, i).get('name'))
            print("-------------------------------------------------------------")

            self.device_index = int(input())
            print(f"Device index: {self.device_index}")

        self.time_start = time.time()
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      input_device_index = self.device_index,
                                      frames_per_buffer=self.CHUNK,
                                      stream_callback=self.callback)

    def callback(self, in_data, frame_count, time_info, status):
        with self._lock:
            self._data.append(in_data)
            if len(self._data) > self.max_buf_len:
                self._data = self._data[-self.max_buf_len:]
            self._event.set()
        return None, pyaudio.paContinue

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def save_test_wav(self):
        wave_file = wave.open("output.wav", 'wb')
        wave_file.setnchannels(self.CHANNELS)
        wave_file.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wave_file.setframerate(self.RATE)
        with self._lock:
            wave_file.writeframes(b''.join(self._data))
        wave_file.close()

    def pull_data(self, blocking=True):
        if blocking:
            self._event.wait()
        with self._lock:
            res = self._data
            self._data = []
        return res
