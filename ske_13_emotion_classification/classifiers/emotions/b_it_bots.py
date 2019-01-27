from enum import Enum

import cv2 as cv
import numpy as np
from numpy import argsort
from keras.models import load_model

from ske_13_emotion_classification.classifiers.base_classifier import BaseClassifier

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def reshape_image_for_keras_model(image, image_size=None):
    if image_size is None:
        image_size = list(image.shape)

    image = rescale(image)
    image = image.reshape(*image_size, 1)

    return image


def rescale(image, scale=1./255):
    return image * scale

# B-IT-BOTS robotics team
# https://github.com/oarriaga/face_classification


class BITBOTS(BaseClassifier):
    def __init__(self, model_path='./data/b_it_bots/fer2013_mini_XCEPTION.119-0.65.hdf5'):
        model = load_model(model_path, compile=False)
        image_size = model.input.shape.as_list()[1:-1]

        self.__model__ = model
        self.__image_size__ = image_size

    def __preprocess__(self, image):
        image = cv.resize(image, tuple(self.__image_size__))
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = reshape_image_for_keras_model(
            image=image,
            image_size=self.__image_size__,
        )

        return image

    def __open_image__(self, filename):
        return cv.imread(filename=filename, flags=0)

    def predict(self, images, verbose=0):
        # if type(images[0]) == 'str':
            # images = [self.__open_image__(filename=image) for image in images]

        images = np.array([self.__preprocess__(image=image)
                           for image in images])
        predictions = self.__model__.predict(images, verbose=verbose)
        results = []
        for p in predictions:
            prediction_indexes = list(reversed(np.argsort(p)))
            result = [[emotions[i], p[i]] for i in prediction_indexes]
            results.append(result)

        return results
