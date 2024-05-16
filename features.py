import numpy as np
from typing import Tuple

from interfaces import ItemFeature, ModelFeature, ComparisonFeature, Model
from dataset import Dataset


class Category(ItemFeature):
    pass


class OptionLogLikelihood(ModelFeature):

    @classmethod
    def compute(cls, dataset: Dataset, model: Model, **kwargs):
        log_likelihoods = model.compute_option_log_likelihoods(items=dataset.items, **kwargs)

        return cls(model=model.name, values=log_likelihoods)


class ModelChoices(ModelFeature):

    @classmethod
    def compute(cls, log_likelihoods: OptionLogLikelihood):
        model_choices = np.argmax(log_likelihoods.values, axis=1)
        return cls(model=log_likelihoods.model, values=model_choices)


class PredictionCorrectness(ModelFeature):

    @classmethod
    def compute(cls, dataset: Dataset, model_choices: ModelChoices):
        ground_truth = np.array([item.correct_index for item in dataset.items])

        correctness = (model_choices.values==ground_truth)*1.
        return cls(model=model_choices.model, values=correctness)


class LogDisagreement(ComparisonFeature):

    @classmethod
    def compute(cls, log_likelihoods: Tuple[OptionLogLikelihood, OptionLogLikelihood]):
        assert len(log_likelihoods)==2

        log_likelihoods1 = log_likelihoods[0].values
        log_likelihoods2 = log_likelihoods[1].values
        model_names = [log_likelihoods[0].model, log_likelihoods[1].model]

        # We want to gather max entries as described in https://stackoverflow.com/a/58098299
        choices1 = np.expand_dims(np.argmax(log_likelihoods1, axis=-1), axis=-1)
        choices2 = np.expand_dims(np.argmax(log_likelihoods2, axis=-1), axis=-1)

        # Generalizing to non-boolean case: Take argmax of each model and use these indices
        model1_selections1 = np.take_along_axis(log_likelihoods1, choices1, axis=-1).squeeze()
        model1_selections2 = np.take_along_axis(log_likelihoods1, choices2, axis=-1).squeeze()
        model2_selections1 = np.take_along_axis(log_likelihoods2, choices1, axis=-1).squeeze()
        model2_selections2 = np.take_along_axis(log_likelihoods2, choices2, axis=-1).squeeze()

        log_disagreement = -1 * (model1_selections1 - model1_selections2) * (model2_selections1 - model2_selections2)
        return cls(models=model_names, values=log_disagreement)
