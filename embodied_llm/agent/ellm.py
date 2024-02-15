from argparse import ArgumentParser
from threading import Lock, Thread
import struct
import re
import time

import cv2
from RealtimeTTS import TextToAudioStream
# from RealtimeSTT import AudioToTextRecorder
import zenoh
import numpy as np

from embodied_llm.asr.real_time_tts import PiperEngine, AudioToTextRecorder


TRIGGER_MSGS = {
    'No_cmd': 0,
    'Turnoff_cmd': 21,
    'Explore_cmd': 1,
    'Home_cmd': 2,
    'Wp_cmd': 3,
    'Find_object_cmd': 4,
    'Sit_cmd': 5,
    'Stand_cmd': 6,
    'Dance_cmd': 7,
    'all_robots_explore_cmd': 8,
    'all_robots_home_cmd': 9,
}


class EmbodiedLLM:
    def __init__(self,
                 input_device,
                 camera_device=0,
                 models_folder=None,
                 pipeline="huggingface",
                 names=("bbeca", "becca", "rebeca", "bekah", "rubika"),
                 zenoh_topic_commands="mist/ellm",
                 zenoh_topic_images="mist/images",
                 zenoh_id=1,
                 remote_camera=False,
                 send_commands=False):

        self.send_commands = send_commands
        self.int_id = zenoh_id
        self.pipline = pipeline
        self.names = names
        self.keep_history = False

        self.recorder = AudioToTextRecorder(model="tiny.en",
                                            language="en",
                                            input_device_index=input_device,
                                            spinner=False)

        if pipeline == "huggingface":
            print(f"DEBUG: using hugging face pipeline")
            from embodied_llm.llm.hugging_face import ImageLLMHuggingFace
            self.llm = ImageLLMHuggingFace(camera_device=camera_device, models_folder=models_folder)
        elif pipeline == "llamacpp":
            print(f"DEBUG: using llama cpp pipeline")
            from embodied_llm.llm.llama_cpp import ImageLLMLlamaCPP
            self.llm = ImageLLMLlamaCPP(camera_device=camera_device, models_folder=models_folder)
        else:
            raise NotImplementedError(pipeline)

        self.searched_str = "baby yoda"
        self._text_buffer = []
        self._lock = Lock()
        self._listening = False
        self.t_listen = None

        self.tts_engine = PiperEngine(models_folder=models_folder)
        self.tts = TextToAudioStream(self.tts_engine, log_characters=False)

        self.remote_camera = remote_camera
        self._image = None
        self._lock_image = Lock()

        self.zenoh_session = zenoh.open()
        self.zenoh_pub = self.zenoh_session.declare_publisher(zenoh_topic_commands)
        self.zenoh_sub = self.zenoh_session.declare_subscriber(zenoh_topic_images, self.receive_zenoh_image)

        time.sleep(2.0)
        self.tts.feed("I'm ready.").play()

    def receive_zenoh_image(self, msg):
        b_string = msg.payload
        array = np.fromstring(b_string, np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        with self._lock_image:
            self._image = image

    def get_image(self):
        cc = True
        while cc:
            with self._lock_image:
                img = self._image
            if img is None:
                print("No received image")
                time.sleep(0.1)
            else:
                cc = False
        return img

    def publish_zenoh_msg(self, int_msg):
        if self.send_commands:
            b_string = struct.pack('H', self.int_id)
            b_string += struct.pack('H', int_msg)
            data = zenoh.Value(b_string)
            self.zenoh_pub.put(data)

    def triggers(self, text):
        """
        Trigger detection.

        Modify this method to implement new triggers, then handle them in the loop method.

        :param text: str: incoming speech transcribed to text
        :return: int: trigger
        """

        name_detected = False
        if any(x.lower() in text.lower() for x in self.names):
            name_detected = True

        trigger = 0

        while True:

            matches = ["bye"]
            if any(x.lower() in text.lower() for x in matches):
                trigger = 1
                break

            matches = ["memoriz", "you", "see"]
            if all(x.lower() in text.lower() for x in matches) and self.pipline == "huggingface":
                trigger = 2
                break

            matches = ["compare"]
            if any(x.lower() in text.lower() for x in matches) and self.pipline == "huggingface":
                trigger = 3

            matches = ["look for", "search"]
            if any(x.lower() in text.lower() for x in matches):
                trigger = 4

            matches = ["sit down"]
            if any(x.lower() in text.lower() for x in matches):
                trigger = 5

            matches = ["stand up"]
            if any(x.lower() in text.lower() for x in matches):
                trigger = 6

            matches = ["describe", "see"]
            if any(x.lower() in text.lower() for x in matches) or self.pipline == "huggingface":
                trigger = 7

            matches = ["reset", "chat"]
            if all(x.lower() in text.lower() for x in matches):
                trigger = 8

            matches = ["history"]
            if any(x.lower() in text.lower() for x in matches):
                matches = ["keep", "track"]
                if any(x.lower() in text.lower() for x in matches):
                    trigger = 9
                matches = ["discard", "do not", "don't", "stop"]
                if any(x.lower() in text.lower() for x in matches):
                    trigger = 10

            break

        return trigger, name_detected

    def listen(self):
        """
        When this method is called, the LLM starts listening in a background thread.

        (Useful for stopping the search mode asynchronously)
        """
        self.t_listen = Thread(target=self._t_listen, args=(), daemon=True)
        self.t_listen.start()

    def _t_listen(self):
        with self._lock:
            self._listening = True
        cc = True
        while cc:
            text = self.recorder.text()
            print(f"DEBUG: listening: {text}")
            with self._lock:
                if self._listening:
                    self._text_buffer.append(text)
                else:
                    cc = False

    def stop_listening(self):
        """
        Stops the listening thread

        (CAUTION: must be called before going back to synchronous mode)
        """
        with self._lock:
            self._listening = False
        if not self.recorder.is_recording:
            self.recorder.start()  # otherwise it blocks
        if self.t_listen is not None:
            self.t_listen.join()
        self.t_listen = None
        with self._lock:
            self._text_buffer = []

    def loop(self, max_iterations=-1):
        """
        Main loop.

        Triggers and different modes are handled here.

        :param max_iterations: int: number of maximum iterations of the loop. -1 for infinite.
        """
        iteration = 0
        mode = "chat"
        while max_iterations < 0 or iteration <= max_iterations:
            if mode == "chat":

                text = self.recorder.text()
                print(f"User: {text}")

                trigger, name_detected = self.triggers(text)
                print(f"DEBUG: name detected: {name_detected}, trigger: {trigger}")
                # if not name_detected:
                #     continue

                if name_detected:

                    if not self.keep_history:
                        self.llm.reset_chat()

                    if trigger == 1:
                        # User said "Bye"
                        self.tts.feed("Goodbye.").play()
                        self.publish_zenoh_msg(TRIGGER_MSGS['Turnoff_cmd'])
                        break
                    elif trigger == 2:
                        # User asked to memorize what the camera currently sees
                        if self.remote_camera:
                            _ = self.get_image()
                            res = "OK"
                        else:
                            res = self.llm.capture_image_and_memorize()
                    elif trigger == 3:
                        # User asked to compare current image with memorized image
                        if self.remote_camera:
                            res = "Sorry, not implemented."
                        else:
                            res = self.llm.capture_image_and_compared_with_memorized(text)
                    elif trigger == 4:
                        # User asked to look for something
                        low = text.lower()
                        idx = low.rindex("look for") + 8
                        while idx < len(low) and not low[idx].isalpha():
                            idx += 1
                        low = low[idx:]
                        low = re.sub('[^a-z ]', '', low)
                        self.searched_str = low
                        res = f"OK, I will look for {self.searched_str}."
                        mode = "search"
                        self.listen()
                        self.publish_zenoh_msg(TRIGGER_MSGS['Explore_cmd'])
                    elif trigger == 5:
                        # User asked to sit down
                        self.publish_zenoh_msg(TRIGGER_MSGS['Sit_cmd'])
                        res = "OK."
                    elif trigger == 6:
                        # User asked to stand up
                        self.publish_zenoh_msg(TRIGGER_MSGS['Stand_cmd'])
                        res = "OK."
                    elif trigger == 7:
                        # User asked to describe
                        if self.remote_camera:
                            image = self.get_image()
                            res = self.llm.image_and_prompt(image, text)
                        else:
                            res = self.llm.capture_image_and_prompt(text)
                    elif trigger == 8:
                        # User asked to reset chat
                        self.llm.reset_chat()
                        res = "I forgot the chat history."
                    elif trigger == 9:
                        # User asked to keep track of chat history
                        self.keep_history = True
                        res = "OK, I'll keep track of chat history."
                    elif trigger == 10:
                        # User asked to not keep track of chat history
                        self.keep_history = False
                        res = "OK, I won't keep track of chat history."
                    else:
                        res = self.llm.simple_prompt(text)

                    print(f"Laika: {res}")
                    self.tts.feed(res).play()
            elif mode == "search":
                self.llm.reset_chat()
                text = f"Do you see {self.searched_str}?"
                if self.remote_camera:
                    image = self.get_image()
                    res = self.llm.image_and_prompt(image, text)
                else:
                    res = self.llm.capture_image_and_prompt(text)

                while len(res) > 0 and not res[0].isalpha():
                    res = res[1:]

                if res.lower().startswith("yes"):
                    # TODO: Found object, trigger action here
                    print(f"DEBUG YES: {res}")
                    idx = 3
                    while idx < len(res) and not res[idx].isalpha():
                        idx += 1
                    if idx >= len(res):
                        res = f"I found {self.searched_str}"
                    else:
                        res = res[idx:]
                    self.tts.feed(res).play()
                    self.stop_listening()  # FIXME: at this point the model needs to hear something, e.g., itself
                    mode = "chat"
                    self.publish_zenoh_msg(TRIGGER_MSGS['Home_cmd'])
                else:
                    stop = False
                    with self._lock:
                        for buf_text in self._text_buffer:
                            if "stop" in buf_text.lower():
                                stop = True
                        self._text_buffer = []
                    if stop:
                        res = f"Fine, I stop looking for {self.searched_str}."
                        self.tts.feed(res).play()
                        self.stop_listening()  # FIXME: at this point the model needs to hear something, e.g., itself
                        mode = "chat"
                        self.publish_zenoh_msg(TRIGGER_MSGS['Home_cmd'])
                    print(f"DEBUG NO: {res}")
            iteration += 1

    def stop(self):
        self.recorder.stop()
        self.recorder.shutdown()
        self.tts.stop()
        self.zenoh_sub.undeclare()
        self.zenoh_session.close()


def main(args):
    from pathlib import Path
    remote = args.remote
    send_commands = args.send_commands
    print(f"REMOTE: {remote}")
    microphone = args.microphone
    camera = args.camera
    max_iterations = args.max_iterations
    pipeline = args.pipeline
    models_folder = args.models_folder
    if models_folder is None:
        models_folder = Path.home() / "ellm"

    ellm = EmbodiedLLM(input_device=microphone, models_folder=models_folder, pipeline=pipeline, camera_device=camera, remote_camera=remote, send_commands=send_commands)
    ellm.loop(max_iterations=max_iterations)
    ellm.stop()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--microphone', type=int, default=-1, help='microphone index')
    parser.add_argument('--camera', type=int, default=-1, help='camera index')
    parser.add_argument('--max-iterations', type=int, default=-1, help='microphone index')
    parser.add_argument('--models-folder', type=str, default=None, help='path of the folder where models should be stored')
    parser.add_argument('--pipeline', type=str, default="llamacpp", help='one of: huggingface, llamacpp')
    parser.add_argument("--remote", help="remote camera", action="store_true")
    parser.add_argument("--send-commands", help="send zenoh commands", action="store_true")
    arguments = parser.parse_args()

    main(arguments)
