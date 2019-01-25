import cv2
import sys
import datetime as dt
from time import sleep
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows

from face_detectors import CascadeFaceDetector, CascadeXMLEnum
from utils import draw_bounding_boxes


def on_failed_to_load_camera():
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass


if __name__ == '__main__':
    face_detector = CascadeFaceDetector(
        xml=CascadeXMLEnum.LBPCASCADE_FRONTALFACE_IMPROVED.value
    )

    video_capture = VideoCapture(0)

    while True:
        on_failed_to_load_camera()

        _, frame = video_capture.read()

        face_detector.detect(
            image=frame,
            then={draw_bounding_boxes}
        )

        if waitKey(1) & 0xFF == ord('q'):
            break

        imshow('Emotion Classification', frame)

    video_capture.release()
    destroyAllWindows()
