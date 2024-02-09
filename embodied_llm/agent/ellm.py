from argparse import ArgumentParser

from RealtimeTTS import TextToAudioStream
from RealtimeSTT import AudioToTextRecorder

from embodied_llm.asr.real_time_tts import PiperEngine
from embodied_llm.llm.llm import ImageLLM


class EmbodiedLLM:
    def __init__(self,
                 input_device,
                 camera_device=0,
                 models_folder=None):

        self.recorder = AudioToTextRecorder(model="tiny.en",
                                            language="en",
                                            input_device_index=input_device,
                                            spinner=False)

        self.llm = ImageLLM(camera_device=camera_device, models_folder=models_folder)
        self.tts_engine = PiperEngine(models_folder=models_folder)
        self.tts = TextToAudioStream(self.tts_engine,
                                     log_characters=False)

        self.tts.feed("I'm ready.").play()

    def triggers(self, text):
        if "bye" in text.lower():
            return 1
        if "what" in text.lower():
            if "you" in text.lower():
                if "see" in text.lower():
                    return 2
        return 0

    def loop(self, max_iterations=-1):
        iteration = 0
        while max_iterations < 0 or iteration <= max_iterations:
            text = self.recorder.text()
            print(f"Speech: {text}")
            trigger = self.triggers(text)
            if trigger == 1:
                self.tts.feed("Goodbye.").play()
                break
            else:
                res = self.llm.capture_image_and_prompt(text)
            # else:
            #     res = self.llm.simple_prompt(text)
            print(f"Laika: {res}")
            self.tts.feed(res).play()
            iteration += 1


def main(args):
    from pathlib import Path
    microphone = args.microphone
    max_iterations = args.max_iterations
    models_folder = args.models_folder
    if models_folder is None:
        models_folder = Path.home() / "ellm"

    # if microphone is None:
    #     import pyaudio
    #
    #     audio = pyaudio.PyAudio()
    #
    #     print("-----------------------Select a microphone-----------------------")
    #     info = audio.get_host_api_info_by_index(0)
    #     num_devices = info.get('deviceCount')
    #     for i in range(0, num_devices):
    #         if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #             print("Input Device id ", i, " - ",
    #                   audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    #     print("-----------------------------------------------------------------")
    #
    #     microphone = int(input())
    #     print(f"Microphone: {microphone}")

    ellm = EmbodiedLLM(input_device=microphone, models_folder=models_folder)
    ellm.loop(max_iterations=max_iterations)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--microphone', type=int, default=-1, help='microphone index')
    parser.add_argument('--max-iterations', type=int, default=-1, help='microphone index')
    parser.add_argument('--models-folder', type=str, default=None, help='path of the folder where models should be stored')
    arguments = parser.parse_args()

    main(arguments)
