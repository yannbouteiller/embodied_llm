import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import pyaudio
import wave
import time
import numpy as np

from embodied_llm.asr.microphone import Microphone


class RealTimeASR:
    def __init__(self,
                 device_index=None,
                 model_id="openai/whisper-large-v3",
                 language="english",
                 model_folder=None,  # None by default (dowloads the model from Hugging Face)
                 end_of_sentence_timer=1.0):

        if model_folder is None:
            model_folder = model_id

        self.language = language

        self.model_folder = model_folder
        self.model_id = model_id

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"using device {self.device} and torch dtype {self.torch_dtype}")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_folder,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={"language": self.language}
        )

        self.mic = Microphone(device_index)
        self.data = []
        self.end_of_sentence_timer = end_of_sentence_timer

    def start(self):
        self.mic.start()

    def stop(self):
        self.mic.stop()

    def reset(self):
        _ = self.mic.pull_data(blocking=False)

    def asr(self, timeout=0, sensitivity=1000, silent_buffer_duration=1.0):  # blocking call
        self.data = []
        temp_result = None
        prev_len = len(self.data)
        ts = None
        time_start = time.time()
        buf_size = int(silent_buffer_duration * self.mic.RATE / self.mic.CHUNK)
        recording_started = False
        while True:
            time.sleep(0.1)
            sample = self.mic.pull_data(blocking=True)
            if len(sample) > 0:
                self.data += sample
                bytestr = b''.join(self.data)
                npstr = np.frombuffer(bytestr, np.int16).astype(np.int32)
                range = np.max(npstr) - np.min(npstr)
                if range < sensitivity and not recording_started:
                    if len(self.data) > buf_size:
                        self.data = self.data[-buf_size:]
                else:  # sound detected
                    recording_started = True
                    waveFile = wave.open("output.wav", 'wb')
                    waveFile.setnchannels(self.mic.CHANNELS)
                    waveFile.setsampwidth(self.mic.audio.get_sample_size(self.mic.FORMAT))
                    waveFile.setframerate(self.mic.RATE)
                    waveFile.writeframes(bytestr)
                    waveFile.close()
                    self.pipe.call_count = 0  # Hack to suppress the warning which asks to use a Dataset
                    result = self.pipe("output.wav")
                    if len(result["text"]) > 0:
                        if temp_result is not None:
                            now = time.time()
                            if result["text"] == temp_result["text"]:
                                if ts is None:
                                    ts = now
                                elif now - ts > self.end_of_sentence_timer:
                                    self.data = []
                                    return result
                            else:
                                ts = now
                            temp_result = result
                        else:
                            temp_result = result
            if timeout > 0:
                if time.time() - time_start > timeout:
                    return None

    def text(self, timeout=0, sensitivity=1500):
        result = self.asr(timeout=timeout, sensitivity=sensitivity)
        return result["text"] if result is not None else None