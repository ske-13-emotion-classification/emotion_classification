from abc import ABC, abstractmethod
from typing import Iterable, Any


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, x: Iterable[Any], verbose: int = 0) -> Iterable[Any]:
        pass
