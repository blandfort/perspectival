import abc
import numpy as np
from typing import Tuple, Any, List

from dataset import Item


class Feature(metaclass=abc.ABCMeta):
    values: Tuple[Any]

    def __init__(self, values: np.array):
        self.values = np.array(values)

    def select(self, indices: List[int]):
        selected_values = self.values[indices]
        return self.__class__(values=selected_values)

    def _get_name(self):
        return self.__class__.__name__

    name = property(fget=_get_name)


class ItemFeature(Feature):
    """Class to store item-based features"""
    pass


class ModelFeature(Feature):
    model: str

    def __init__(self, values, model: str):
        super().__init__(values)
        self.model = model

    def select(self, indices):
        selected_values = self.values[indices]
        return self.__class__(values=selected_values, model=self.model)

    @abc.abstractmethod
    def compute(cls):
        raise NotImplementedError()


class ComparisonFeature(Feature):
    models: Tuple[str, str]

    def __init__(self, values, models: Tuple[str, str]):
        super().__init__(values)
        self.models = tuple(sorted(list(models)))

    def select(self, indices):
        selected_values = self.values[indices]
        return self.__class__(values=selected_values, models=self.models)

    @abc.abstractmethod
    def compute(cls):
        raise NotImplementedError()


class Model(metaclass=abc.ABCMeta):
    name: str

    @abc.abstractmethod
    def compute_option_log_likelihoods(self, items: List[Item]):
        raise NotImplementedError()
