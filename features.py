import abc
import numpy as np
from typing import Tuple, Any, Type

from model import Model


class Feature(metaclass=abc.ABCMeta):
    name: str
    values: Tuple[Any]

    def __init__(self, values):
        self.values = np.array(values)

    @classmethod
    def key(cls, **ignored):
        return cls.name

    def select(self, indices):
        selected_values = np.take(self.values, indices)
        return self.__class__(values=selected_values)

    def compute(self, dataset):
        raise NotImplementedError()


class CategoryFeature(Feature):
    name = 'Category'


class Prediction(Feature):
    model: str

    def __init__(self, values, model: str):
        super().__init__(values)
        self.model = model

    @classmethod
    def key(cls, model: Model, **ignored):
        return (cls.name, model.name)

    def select(self, indices):
        selected_values = np.take(self.values, indices)
        return self.__class__(values=selected_values, model=self.model)

class ModelChoices(Prediction):
    name = 'Model Choices'

    @classmethod
    def compute(cls, dataset, model: Model):
        dataset.ensure_feature(OptionLogLikelihood, model=model)
        log_likelihoods = dataset.get_feature(OptionLogLikelihood, model=model).values

        model_choices = np.argmax(log_likelihoods, axis=1)
        return cls(model=model.name, values=model_choices)

class PredictionCorrectness(Prediction):
    name = 'Prediction Correctness'

    @classmethod
    def compute(cls, dataset, model: Model):
        dataset.ensure_feature(ModelChoices, model=model)
        model_choices = dataset.get_feature(ModelChoices, model=model).values
        ground_truth = np.array([item.correct_index for item in dataset.items])

        correctness = (model_choices==ground_truth)*1.
        return cls(model=model.name, values=correctness)


class OptionLogLikelihood(Prediction):
    name = 'Option Log Likelihood'

    @classmethod
    def compute(cls, dataset, model: Model):
        log_likelihoods = []

        for item in dataset.items:
            item_log_likelihoods = model.compute_option_log_likelihoods(item)
            log_likelihoods.append(item_log_likelihoods)
        values = np.array(log_likelihoods)
        return cls(model=model.name, values=values)


class Comparison(Feature):
    models: Tuple[str, str]

    def __init__(self, values, models: Tuple[str, str]):
        super().__init__(values)
        self.models = tuple(sorted(list(models)))

    @classmethod
    def key(cls, models: Tuple[Model, Model], **ignored):
        return (cls.name, tuple(sorted(list(model.name for model in models))))

    def select(self, indices):
        selected_values = np.take(self.values, indices)
        return self.__class__(values=selected_values, models=self.models)


class LogDisagreement(Comparison):
    name = 'Log Disagreement'

    @classmethod
    def compute(cls, dataset, models: Tuple[Model, Model]):
        assert len(models)==2

        dataset.ensure_feature(OptionLogLikelihood, model=models[0])
        log_likelihoods1 = dataset.get_feature(OptionLogLikelihood, model=models[0]).values
        dataset.ensure_feature(OptionLogLikelihood, model=models[1])
        log_likelihoods2 = dataset.get_feature(OptionLogLikelihood, model=models[1]).values

        # We want to gather max entries as described in https://stackoverflow.com/a/58098299
        choices1 = np.expand_dims(np.argmax(log_likelihoods1, axis=-1), axis=-1)
        choices2 = np.expand_dims(np.argmax(log_likelihoods2, axis=-1), axis=-1)

        # Generalizing to non-boolean case: Take argmax of each model and use these indices
        model1_selections1 = np.take_along_axis(log_likelihoods1, choices1, axis=-1).squeeze()
        model1_selections2 = np.take_along_axis(log_likelihoods1, choices2, axis=-1).squeeze()
        model2_selections1 = np.take_along_axis(log_likelihoods2, choices1, axis=-1).squeeze()
        model2_selections2 = np.take_along_axis(log_likelihoods2, choices2, axis=-1).squeeze()

        log_disagreement = -1 * (model1_selections1 - model1_selections2) * (model2_selections1 - model2_selections2)
        return cls(models=[model.name for model in models], values=log_disagreement)
