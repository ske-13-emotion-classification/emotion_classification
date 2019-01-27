from enum import Enum
from typing import List, Set, Iterable, Callable

from cv2 import CascadeClassifier

from ske_13_emotion_classification.detectors import BaseDetector


class CascadeXMLEnum(Enum):
    HAARCASCADE_FRONTALFACE_DEFAULT = './data/haarcascades/haarcascade_frontalface_default.xml'
    HAARCASCADE_FRONTALFACE_ALT = './data/haarcascades/haarcascade_frontalface_alt.xml'
    HAARCASCADE_FRONTALFACE_ALT2 = './data/haarcascades/haarcascade_frontalface_alt.xml'
    HAARCASCADE_FRONTALFACE_ALT_TREE = './data/haarcascades/haarcascade_frontalface_alt.xml'
    HAARCASCADE_PROFILEFACE = './data/haarcascades/haarcascade_profileface.xml'

    HAARCASCADE_CUDA_FRONTALFACE_DEFAULT = './data/haarcascades_cuda/haarcascade_frontalface_default.xml'
    HAARCASCADE_CUDA_FRONTALFACE_ALT = './data/haarcascades_cuda/haarcascade_frontalface_alt.xml'
    HAARCASCADE_CUDA_FRONTALFACE_ALT2 = './data/haarcascades_cuda/haarcascade_frontalface_alt.xml'
    HAARCASCADE_CUDA_FRONTALFACE_ALT_TREE = './data/haarcascades_cuda/haarcascade_frontalface_alt.xml'
    HAARCASCADE_CUDA_PROFILEFACE = './data/haarcascades_cuda/haarcascade_profileface.xml'

    LBPCASCADE_PROFILEFACE = './data/lbpcascades/lbpcascade_profileface.xml'
    LBPCASCADE_FRONTALFACE = './data/lbpcascades/lbpcascade_frontalface.xml'
    LBPCASCADE_FRONTALFACE_IMPROVED = './data/lbpcascades/lbpcascade_frontalface_improved.xml'


class Cascade(BaseDetector):
    def __init__(self, xml: CascadeXMLEnum = CascadeXMLEnum.HAARCASCADE_FRONTALFACE_DEFAULT.value):
        if type(xml) == type(CascadeXMLEnum.HAARCASCADE_CUDA_FRONTALFACE_ALT2):
            xml = xml.value
        self.__model__ = CascadeClassifier(xml)

    def detect(self, image, then: Iterable[Callable] = None) -> List[int]:
        model = self.__model__

        faces = model.detectMultiScale(
            image=image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if then is not None:
            for function in then:
                function(image, faces)

        return faces
