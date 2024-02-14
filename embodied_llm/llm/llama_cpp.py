import signal
from pathlib import Path
import base64

import cv2
from PIL import Image
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
                 main_gpu=0,
                 n_gpu_layers=-1
                 ):

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

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

        chat_handler = Llava15ChatHandler(clip_model_path=str(path_clip))

        self.llama = Llama(
            model_path=str(path_model),
            chat_handler=chat_handler,
            n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
            logits_all=True,  # needed to make llava work
            main_gpu=main_gpu,
            n_gpu_layers=n_gpu_layers
        )

    def stop(self):
        self.p.send_signal(signal.SIGINT)

    def reset_chat(self):
        pass

    def capture_image_and_prompt(self, text):

        self.cam.grab()
        ret, shot = self.cam.read()
        if ret and shot is not None:
            # display(shot)
            base64_image = base64.b64encode(cv2.imencode('.jpg', shot)[1]).decode('utf-8')
            color_converted = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            self.images_hist.append(pil_image)
        else:
            raise RuntimeError("Something is wrong with the camera")

        response = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant who answers concisely."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {f"type": "text", "text": f"(CONTEXT: imagine you are seeing this image) {text}"}
                    ]
                }
            ]
        )

        text = response['choices'][0]['message']['content']

        # if len(self.prompt) > 0:
        #     self.prompt += "\n"
        # self.prompt += "USER: "


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
        return text

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
        return "Not implemented, sorry."

    def image_and_prompt(self, image, text):
        return "Not implemented, sorry."

    def image_and_prompt(image, text):
        pass

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
