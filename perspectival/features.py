import sys
from typing import Tuple
import numpy as np

from .interfaces import ItemFeature, ModelFeature, ComparisonFeature, Model
from .dataset import Dataset, Item


class Category(ItemFeature):
    values: Tuple[str]


class GroundTruth(ItemFeature):
    values: Tuple[int]


class NamedFeature(ItemFeature):
    name = None

    def __init__(self, values: np.array, name: str):
        super().__init__(values)
        self.name = name

    def select(self, indices):
        selected_values = self.values[indices]
        return self.__class__(values=selected_values, name=self.name)


class OptionLogLikelihood(ModelFeature):

    @classmethod
    def compute(cls, dataset: Dataset, model: Model, **kwargs):
        log_likelihoods = model.compute_option_log_likelihoods(
            items=dataset.items, **kwargs
        )

        # In case different items have different numbers of options, we need padding
        # Determine the maximum length of the inner lists
        max_length = max(len(inner_list) for inner_list in log_likelihoods)

        # Pad the inner lists with np.nan to make them all the same length
        padding_value = -1e9
        padded_log_likelihoods = np.array(
            [
                inner_list + [padding_value] * (max_length - len(inner_list))
                for inner_list in log_likelihoods
            ]
        )

        return cls(model=model.name, values=padded_log_likelihoods)


class ModelChoices(ModelFeature):

    @classmethod
    def compute(cls, log_likelihoods: OptionLogLikelihood):
        model_choices = np.argmax(log_likelihoods.values, axis=1)
        return cls(model=log_likelihoods.model, values=model_choices)


class TextContinuation(ModelFeature):

    @classmethod
    def compute(cls, dataset: Dataset, model: Model, **kwargs):
        continuations = model.compute_continuations(items=dataset.items, **kwargs)
        return cls(model=model.name, values=continuations)


class PredictionCorrectness(ModelFeature):

    @classmethod
    def compute(cls, model_choices: ModelChoices, ground_truth: GroundTruth):
        correctness = (model_choices.values == ground_truth.values) * 1.0
        return cls(model=model_choices.model, values=correctness)


class NamedModelFeature(ModelFeature):
    name = None

    def __init__(self, values: np.array, model: str, name: str):
        super().__init__(values, model)
        self.name = name

    def select(self, indices):
        selected_values = self.values[indices]
        return self.__class__(values=selected_values, model=self.model, name=self.name)

    @classmethod
    def compute(cls):
        raise NotImplementedError(
            "This feature should be instantiated by passing values to __init__!"
        )


def compute_disagreement_score(
    log_likelihoods: Tuple[OptionLogLikelihood, OptionLogLikelihood]
):
    assert len(log_likelihoods) == 2

    log_likelihoods1 = log_likelihoods[0].values
    log_likelihoods2 = log_likelihoods[1].values

    # We want to gather max entries as described in https://stackoverflow.com/a/58098299
    choices1 = np.expand_dims(np.argmax(log_likelihoods1, axis=-1), axis=-1)
    choices2 = np.expand_dims(np.argmax(log_likelihoods2, axis=-1), axis=-1)

    # Generalizing to non-boolean case: Take argmax of each model and use these indices
    model1_selections1 = np.take_along_axis(
        log_likelihoods1, choices1, axis=-1
    ).squeeze(axis=-1)
    model1_selections2 = np.take_along_axis(
        log_likelihoods1, choices2, axis=-1
    ).squeeze(axis=-1)
    model2_selections1 = np.take_along_axis(
        log_likelihoods2, choices1, axis=-1
    ).squeeze(axis=-1)
    model2_selections2 = np.take_along_axis(
        log_likelihoods2, choices2, axis=-1
    ).squeeze(axis=-1)

    # Normalize log likelihoods so that disagreement scores will be in [0, 1]
    # We only want to consider predictions for the relevant options
    selected_log_likelihoods1 = np.stack(
        (model1_selections1, model1_selections2), axis=-1
    )
    selected_log_likelihoods2 = np.stack(
        (model2_selections1, model2_selections2), axis=-1
    )
    n_log_likelihoods1 = np.array(selected_log_likelihoods1) / np.max(
        np.abs(selected_log_likelihoods1), axis=1
    ).reshape(-1, 1).repeat(2, axis=-1)
    n_log_likelihoods2 = np.array(selected_log_likelihoods2) / np.max(
        np.abs(selected_log_likelihoods2), axis=1
    ).reshape(-1, 1).repeat(2, axis=-1)

    # We take square root to make the scores more intuitive
    log_disagreement = np.sqrt(
        -1
        * (n_log_likelihoods1[:, 0] - n_log_likelihoods2[:, 0])
        * (n_log_likelihoods1[:, 1] - n_log_likelihoods2[:, 1])
    )
    return log_disagreement


class ContinuationDisagreement(ComparisonFeature):

    @classmethod
    def compute(
        cls,
        dataset: Dataset,
        continuations: Tuple[TextContinuation, TextContinuation],
        models: Tuple[Model, Model],
        **kwargs
    ):
        assert len(continuations) == 2
        assert len(models) == 2

        # Compute log likelihoods for both continuations
        # First, we create input in Item format so we can invoke the usual functions
        pseudo_items = []
        for ix, item in enumerate(dataset.items):
            options = [continuations[0].values[ix], continuations[1].values[ix]]
            pseudo_items.append(Item(id=item.id, prompt=item.prompt, options=options))

        log_likelihoods1 = OptionLogLikelihood(
            values=models[0].compute_option_log_likelihoods(
                items=pseudo_items, **kwargs
            ),
            model=models[0].name,
        )
        log_likelihoods2 = OptionLogLikelihood(
            models[1].compute_option_log_likelihoods(items=pseudo_items, **kwargs),
            model=models[1].name,
        )
        log_disagreement = compute_disagreement_score(
            log_likelihoods=[log_likelihoods1, log_likelihoods2]
        )
        model_names = [log_likelihoods1.model, log_likelihoods2.model]
        return cls(models=model_names, values=log_disagreement)


class LogDisagreement(ComparisonFeature):

    @classmethod
    def compute(cls, log_likelihoods: Tuple[OptionLogLikelihood, OptionLogLikelihood]):
        log_disagreement = compute_disagreement_score(log_likelihoods=log_likelihoods)
        model_names = [log_likelihoods[0].model, log_likelihoods[1].model]
        return cls(models=model_names, values=log_disagreement)


class BinaryDisagreement(ComparisonFeature):

    @classmethod
    def compute(cls, log_disagreement: LogDisagreement):
        values = log_disagreement.values > 0
        return cls(models=log_disagreement.models, values=values)


def feature_from_dict(d):
    """Utility function to load a feature based on a dictionary"""
    # Use the name to identify the class
    cls_name = d["name"]
    cls = getattr(sys.modules[__name__], cls_name, None)
    if cls is not None:
        assert issubclass(cls, (ItemFeature, ModelFeature, ComparisonFeature))

        # The remaining arguments are contents of the feature
        return cls(**{key: d[key] for key in d if key != "name"})

    if "model" in d:
        cls = NamedModelFeature
    else:
        cls = NamedFeature

    return cls(**{key: d[key] for key in d})
