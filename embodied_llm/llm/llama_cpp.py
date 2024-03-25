import signal
from pathlib import Path
import base64

import cv2
# from PIL import Image
from matplotlib import pyplot as plt

import openai

from embodied_llm.llm.image_llm import ImageLLM
import json
import copy

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
                 main_gpu=0,
                 n_gpu_layers=-1,
                 language="en"
                 ):

        language = language.lower()
        self.language = language

        supported = ["en", "fr"]
        if self.language not in supported:
            raise RuntimeError(f"Unsupported language designation: {self.language}. Supported designations are {supported}")

        # from llama_cpp import Llama
        # from llama_cpp.llama_chat_format import Llava15ChatHandler

        models_folder = Path(models_folder)

        init_str = \
            f"You are Jarvis, a helpful robot. You are in the MIST Robotics Lab in Polytechnique Montreal.\
            The pictures you receive are what you see in front of you. You must answer in 35 words or less." \
            if self.language == "en" else \
            f"Tu es Jarvis, un robot utile. Tu es au MIST, le laboratoire de robotique de Polytechnique Montreal.\
            Les images que tu recois sont ce que tu vois. Tes reponses ne doivent pas depasser 35 mots."

        self.context = [
            {
                "role": "system", 
                "content": init_str
            },
        ]

        self.model_name = model_name
        self.clip_name = clip_name
        self.models_folder = models_folder
        self.camera_device = camera_device
        self.prompt = ""
        self.cam = cv2.VideoCapture(camera_device)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        mp = Path(self.models_folder)
        path_clip = mp / self.clip_name
        path_model = mp / self.model_name

        self.messages = None
        self.max_history = 10 * 2 + 1  # must be odd to keep the parity  
        self.reset_chat()

        if not path_clip.exists():
            raise FileNotFoundError(path_clip)
        if not path_model.exists():
            raise FileNotFoundError(path_model)

        #chat_handler = Llava15ChatHandler(clip_model_path=str(path_clip))

        self.llama = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")

    def stop(self):
        pass

    def reset_chat(self):
        # self.messages = [{"role": "system", "content": f"You are Jarvis, a helpful robot. You are in the MIST Lab at Polytechnique Montreal.  You answer clearly, politely, and concisely."}]
        self.messages = copy.deepcopy(self.context)

    def clip_history(self):
        if len(self.messages) > self.max_history + len(self.context):
            context = copy.deepcopy(self.context)
            hist = self.messages[-self.max_history:]
            self.messages = context + hist
            assert len(self.messages) == len(self.context) + self.max_history
            assert self.messages[len(self.context)]['role'] == 'user'

    def print_messages_no_image(self):
        for message in self.messages:
            printable = copy.deepcopy(message)
            if isinstance(printable['content'], list):
                for content in printable['content']:
                    if content['type'] == "image_url":
                        content['image_url'] = "string"
            print(json.dumps(printable, indent=4))

    def add_message(self, message):
        self.messages.append(message)
        self.clip_history()

    def clean_history(self):
        for message in self.messages:
            if isinstance(message['content'], list):
                img_idx = None
                for i, content in enumerate(message['content']):
                    if content['type'] == "image_url":
                        img_idx = i 
                if img_idx is not None:
                    message['content'].pop(img_idx)

    def capture_image_and_prompt(self, text):
        self.cam.grab()
        ret, shot = self.cam.read()
        if ret and shot is not None:
            # display(shot)
            base64_image = base64.b64encode(cv2.imencode('.png', shot)[1]).decode('utf-8')
            # color_converted = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            # pil_image = Image.fromarray(color_converted)
        else:
            raise RuntimeError("Something is wrong with the camera")

        text_str = \
            f"(The above picture is what you see. Do not in any circumstance refer to it as being an image.) {text}" \
            if self.language == "en" else \
            f"(L'image ci-dessus est ce que tu vois. Ne mentionne en aucun cas le fait que ce soit une image.) {text}"

        new_message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {f"type": "text",
                     "text": text_str}]}

        self.add_message(new_message)

        # self.print_messages_no_image()

        response = self.llama.chat.completions.create(
            stream=True,
            model=self.model_name,
            messages=self.messages
        )

        self.clean_history()

        output = ""
        for completion_chunk in response:
            text = completion_chunk.choices[0].delta.content
            if not text:
                continue
            output += text
            yield text

        print(output)
        new_message = {
            "role": "assistant",
            "content": f"{output}"}
        self.messages.append(new_message)

    def image_and_prompt(self, image, text):

        base64_image = base64.b64encode(cv2.imencode('.png', image)[1]).decode('utf-8')

        text_str = \
            f"(The above picture is what you see. Do not in any circumstance refer to it as being an image.) {text}" \
            if self.language == "en" else \
            f"(L'image ci-dessus est ce que tu vois. Ne mentionne en aucun cas le fait que ce soit une image.) {text}"

        new_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {f"type": "text",
                 "text": text_str}]}

        self.add_message(new_message)

        # self.print_messages_no_image()

        response = self.llama.chat.completions.create(
            stream=True,
            model=self.model_name,
            messages=self.messages
        )
        
        self.clean_history()

        output = ""
        for completion_chunk in response:
            text = completion_chunk.choices[0].delta.content
            if not text:
                continue
            output += text
            yield text

        print(output)
        new_message = {
            "role": "assistant",
            "content": f"{output}"}
        self.messages.append(new_message)

    def simple_prompt(self, text):

        new_message = {
            "role": "user",
            "content": [
                {f"type": "text", "text": f"{text}"}]}

        self.add_message(new_message)

        # self.print_messages_no_image()

        response = self.llama.chat.completions.create(
            model=self.model_name,
            stream=True,
            messages=self.messages
        )

        output = ""
        for completion_chunk in response:
            text = completion_chunk.choices[0].delta.content
            if not text:
                continue
            output += text
            yield text

        print(output)
        new_message = {
            "role": "assistant",
            "content": f"{output}"}
        self.messages.append(new_message)

    def capture_image_and_memorize(self):
        # self.reset_chat()
        # self.prompt += "USER: "
        #
        # self.cam.grab()
        # ret, shot = self.cam.read()
        # if ret and shot is not None:
        #     # display(shot)
        #     color_converted = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
        #     pil_image = Image.fromarray(color_converted)
        #     self.memorized_img = pil_image
        #     self.images_hist.append(pil_image)
        #     self.prompt += "<image>\n"
        #     self.prompt += "Concisely describe what you see in this picture."
        # else:
        #     print(f"WARNING: the camera could not capture an image")
        #
        # self.prompt += "\nASSISTANT:"
        #
        # inputs = self.processor(self.prompt, self.images_hist, return_tensors='pt').to(0, torch.float16)
        # output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        # res = self.processor.decode(output[0], skip_special_tokens=False)
        # res = res[4:-4]
        # self.prompt = res
        # idx = self.prompt.rindex("ASSISTANT: ")
        # res = self.prompt[idx + 11:]
        # return res
        return "Not implemented, sorry."

    def capture_image_and_compared_with_memorized(self, text):
        # self.reset_chat()
        # if self.memorized_img is None:
        #     return "I have not memorized any image, sorry."
        #
        # self.images_hist.append(self.memorized_img)
        #
        # self.prompt += "USER: "
        # self.prompt += "<image>\n"  # first image
        #
        # self.cam.grab()
        # ret, shot = self.cam.read()
        # if ret and shot is not None:
        #     # display(shot)
        #     color_converted = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
        #     pil_image = Image.fromarray(color_converted)
        #     self.images_hist.append(pil_image)
        #     self.prompt += "<image>\n"  # second image
        # else:
        #     print(f"WARNING: the camera could not capture an image")
        #
        # self.prompt += text
        # self.prompt += "\nASSISTANT:"
        #
        # inputs = self.processor(self.prompt, self.images_hist, return_tensors='pt').to(0, torch.float16)
        # output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        # res = self.processor.decode(output[0], skip_special_tokens=False)
        # res = res[4:-4]
        # self.prompt = res
        # idx = self.prompt.rindex("ASSISTANT: ")
        # res = self.prompt[idx + 11:]
        # return res
        return "Not implemented, sorry."
