import cv2
import zenoh
import time
from threading import Thread


class Streamer:
    def __init__(self, camera_index=0, zenoh_topic="mist/images", sleep_duration=1.0):
        self.sleep_duration = sleep_duration
        self.camera_index = camera_index
        self.zenoh_topic = zenoh_topic
        self.cam = cv2.VideoCapture(self.camera_index)
        ret, shot = self.cam.read()
        if not ret or shot is None:
            raise RuntimeError(f"Something went wrong with the camera at index {self.camera_index}")

        self.image_shape = shot.shape

        self.zenoh_session = zenoh.open()
        self.zenoh_pub = self.zenoh_session.declare_publisher(self.zenoh_topic)
        self.zenoh_sub = self.zenoh_session.declare_subscriber(self.zenoh_topic, self.receive_zenoh_msg)

        self._spinning = False
        self._t_spin = None

    def stop(self):
        if self._t_spin is not None:
            self._spinning = False
            self._t_spin.join()

    def destroy(self):
        self.stop()
        self.zenoh_pub.undeclare()
        self.zenoh_sub.undeclare()
        self.zenoh_session.close()

    def receive_zenoh_msg(self, msg):
        pass

    def publish_zenoh_msg(self, image):
        b_string = cv2.imencode('.png', image)[1].tobytes()
        # length = len(b_string)
        # b_string_length = struct.pack('H', length)
        # b_string = b_string_length + b_string
        data = zenoh.Value(b_string)
        self.zenoh_pub.put(data)

    def spin(self):
        if not self._spinning:
            self._spinning = True
            self._t_spin = Thread(target=self._spin, args=(), daemon=True)
            self._t_spin.start()

    def _spin(self):
        while True:
            self.cam.grab()
            ret, shot = self.cam.read()
            if ret and shot is not None:
                self.publish_zenoh_msg(shot)
            else:
                raise RuntimeError(f"Something went wrong with the camera at index {self.camera_index}")
            if not self._spinning:
                break
            else:
                time.sleep(self.sleep_duration)


if __name__ == "__main__":
    st = Streamer(camera_index=4,
                  zenoh_topic="mist/images")
    st.spin()

    time.sleep(30)

    st.destroy()
