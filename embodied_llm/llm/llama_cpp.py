import signal
from pathlib import Path
import subprocess

import cv2
import torch
from matplotlib import pyplot as plt

from embodied_llm.llm.image_llm import ImageLLM


def display(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title('my picture')
    plt.show()


class ImageLLMLlamaCPP(ImageLLM):
    def __init__(self,
                 models_folder,
                 model_name="ggml-model-q4_k.gguf",
                 clip_name="mmproj-model-f16.gguf",
                 camera_device=-1,
                 main_gpu=None,
                 n_gpu_layers=-1
                 ):

        models_folder = Path(models_folder)

        self.model_name = model_name
        self.clip_name = clip_name
        self.models_folder = models_folder
        self.camera_device = camera_device
        self.images_hist = []
        self.prompt = ""
        self.cam = cv2.VideoCapture(camera_device)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        mp = Path(self.models_folder)
        path_clip = mp / self.clip_name
        path_model = mp / self.model_name

        if not path_clip.exists():
            raise FileNotFoundError(path_clip)
        if not path_model.exists():
            raise FileNotFoundError(path_model)

        # starting server

        command = f"python3 -m llama_cpp.server --model {path_clip} --clip_model_path {path_model} --chat_format llava-1-5"
        if main_gpu is not None:
            command += f" --main_gpu {main_gpu} --n_gpu_layers {n_gpu_layers}"

        self.p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    def stop(self):
        self.p.send_signal(signal.SIGINT)

    def reset_chat(self):
        pass

    def capture_image_and_prompt(self, text):
        # if len(self.prompt) > 0:
        #     self.prompt += "\n"
        # self.prompt += "USER: "
        # self.cam.grab()
        # ret, shot = self.cam.read()
        # if ret and shot is not None:
        #     # display(shot)
        #     color_converted = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
        #     pil_image = Image.fromarray(color_converted)
        #     self.prompt += "<image>\n"
        #     self.prompt += "(CONTEXT: imagine you are seeing this image) "
        #     self.images_hist.append(pil_image)
        # else:
        #     print(f"WARNING: the camera could not capture an image")
        #
        # self.prompt += text + "\nASSISTANT:"
        #
        # inputs = self.processor(self.prompt, self.images_hist, return_tensors='pt').to(0, torch.float16)
        # output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        # res = self.processor.decode(output[0], skip_special_tokens=False)
        # res = res[4:-4]
        # self.prompt = res
        # idx = self.prompt.rindex("ASSISTANT: ")
        # res = self.prompt[idx + 11:]
        return "The llama-CPP pipeline is not implemented, sorry."

    def simple_prompt(self, text):
        # if len(self.prompt) > 0:
        #     self.prompt += "\n"
        # self.prompt += "USER: "
        # self.prompt += text + "\nASSISTANT:"
        #
        # inputs = self.processor(self.prompt, return_tensors='pt').to(0, torch.float16)
        # output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        # res = self.processor.decode(output[0], skip_special_tokens=False)
        # res = res[4:-4]
        # self.prompt = res
        # idx = self.prompt.rindex("ASSISTANT: ")
        # res = self.prompt[idx + 11:]
        return "The llama-CPP pipeline is not implemented, sorry."

    def image_and_prompt(self, image, text):
        return "The llama-CPP pipeline is not implemented, sorry."
