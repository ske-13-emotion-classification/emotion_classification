from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image, then=[]):
        pass
