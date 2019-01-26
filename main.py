from time import sleep

from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
import cv2

from emotion_classifiers import BITBOTSEmotionClassifier
from face_detectors import CascadeFaceDetector, CascadeXMLEnum
from utils import draw_bounding_boxes, draw_texts, extract_objects


def on_failed_to_load_camera():
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass


if __name__ == '__main__':
    face_detector = CascadeFaceDetector(
        xml=CascadeXMLEnum.HAARCASCADE_FRONTALFACE_DEFAULT)
    emotion_classifier = BITBOTSEmotionClassifier()

    video_capture = VideoCapture(0)

    while True:
        on_failed_to_load_camera()

        _, frame = video_capture.read()

        face_bounding_boxes = face_detector.detect(image=frame)

        draw_bounding_boxes(
            image=frame,
            bouding_boxes=face_bounding_boxes
        )

        faces = extract_objects(
            image=frame,
            bounding_boxes=face_bounding_boxes,
        )

        if len(faces) != 0:
            results = emotion_classifier.predict(images=faces, verbose=1)
            for r, bbox in zip(results, face_bounding_boxes):
                cv2.putText(
                    frame,
                    str(r[0][0]),
                    (bbox[0], bbox[-1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .5,
                    (255, 255, 255),
                    lineType=cv2.LINE_AA
                )

        if waitKey(1) & 0xFF == ord('q'):
            break

        imshow('Emotion Classification', frame)

    video_capture.release()
    destroyAllWindows()
