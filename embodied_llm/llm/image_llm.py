from abc import ABC


class ImageLLM(ABC):

    def reset_chat(self):
        raise NotImplementedError

    def capture_image_and_prompt(self, text):
        raise NotImplementedError

    def prompt(self, text):
        raise NotImplementedError

    def image(self, image):
        raise NotImplementedError

    def capture_image_and_memorize(self):
        raise NotImplementedError

    def capture_image_and_compared_with_memorized(self, text):
        raise NotImplementedError
