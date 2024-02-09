import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pathlib import Path

from matplotlib import pyplot as plt


def display(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title('my picture')
    plt.show()


class ImageLLM:
    def __init__(self,
                 models_folder,
                 model_id="llava-hf/bakLlava-v1-hf",
                 camera_device=-1):

        model_folder = Path(models_folder) / "bakllava"

        self.model_id = model_id
        self.model_folder = model_folder
        self.camera_device = camera_device
        self.images_hist = []
        self.prompt = ""
        self.cam = cv2.VideoCapture(camera_device)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        mp = Path(self.model_folder)
        if mp.exists():
            # load model
            self.model = LlavaForConditionalGeneration.from_pretrained(self.model_folder,
                                                                       torch_dtype=torch.float16,
                                                                       low_cpu_mem_usage=True,
                                                                       load_in_4bit=True)
        else:
            # download and save model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_4bit=True  # TODO: change this to new bitsandbytes config
            )
            self.model.save_pretrained(self.model_folder)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def reset_chat(self):
        self.images_hist = []
        self.prompt = ""

    def capture_image_and_prompt(self, text):
        if len(self.prompt) > 0:
            self.prompt += "\n"
        self.prompt += "USER: "
        self.cam.grab()
        ret, shot = self.cam.read()
        if ret and shot is not None:
            # display(shot)
            color_converted = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            self.prompt += "<image>\n"
            self.prompt += "(CONTEXT: imagine you are seeing this image) "
            self.images_hist.append(pil_image)
        else:
            print(f"WARNING: the camera could not capture an image")

        self.prompt += text + "\nASSISTANT:"

        inputs = self.processor(self.prompt, self.images_hist, return_tensors='pt').to(0, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        res = self.processor.decode(output[0], skip_special_tokens=False)
        res = res[4:-4]
        self.prompt = res
        idx = self.prompt.rindex("ASSISTANT: ")
        res = self.prompt[idx + 11:]
        return res

    def simple_prompt(self, text):
        if len(self.prompt) > 0:
            self.prompt += "\n"
        self.prompt += "USER: "
        self.prompt += text + "\nASSISTANT:"

        inputs = self.processor(self.prompt, return_tensors='pt').to(0, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        res = self.processor.decode(output[0], skip_special_tokens=False)
        res = res[4:-4]
        self.prompt = res
        idx = self.prompt.rindex("ASSISTANT: ")
        res = self.prompt[idx + 11:]
        return res

    def image_and_prompt(image, text):
        pass
