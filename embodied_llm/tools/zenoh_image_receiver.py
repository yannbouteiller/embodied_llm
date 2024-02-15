import time

import cv2
import zenoh
from threading import Lock

import numpy as np


class Receiver:
    def __init__(self, zenoh_topic_images="mist/images"):
        self.zenoh_session = zenoh.open()
        self.zenoh_sub = self.zenoh_session.declare_subscriber(zenoh_topic_images, self.receive_zenoh_image)
        self._image = None
        self._lock_image = Lock()

    def receive_zenoh_image(self, msg):
        import struct
        print(f"RECEIVED IMAGE")
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


if __name__ == "__main__":
    re = Receiver()
    img = re.get_image()
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
