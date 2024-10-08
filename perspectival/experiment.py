from typing import List, Union, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from joblib import dump, load

from .interfaces import Feature, ItemFeature, ModelFeature, ComparisonFeature, Model
from .dataset import Dataset
from .features import (
    OptionLogLikelihood,
    ModelChoices,
    PredictionCorrectness,
    LogDisagreement,
    BinaryDisagreement,
    TextContinuation,
    ContinuationDisagreement,
    feature_from_dict,
)


class PreconditionError(Exception):
    """Exception to raise if any precondition to run code is not fulfilled."""


class Experiment:
    name: str
    dataset: Dataset
    item_features: Dict[str, ItemFeature] = {}
    model_features: Dict[str, Dict[str, ModelFeature]] = {}
    comparison_features: Dict[str, Dict[Tuple[str, str], ComparisonFeature]] = {}

    def __init__(
        self,
        name,
        dataset,
        *,
        item_features=None,
        model_features=None,
        comparison_features=None,
        features: Optional[List[Feature]] = None,
    ):
        self.name = name
        self.dataset = dataset

        self.item_features = item_features if item_features is not None else {}
        self.model_features = model_features if model_features is not None else {}
        self.comparison_features = (
            comparison_features if comparison_features is not None else {}
        )

        if features is not None:
            for feature in features:
                self.register_feature(feature)

    def register_feature(self, feature: Feature):
        if isinstance(feature, ItemFeature):
            self.item_features[feature.name] = feature
        elif isinstance(feature, ModelFeature):
            if feature.name not in self.model_features:
                self.model_features[feature.name] = {}
            self.model_features[feature.name][feature.model] = feature
        elif isinstance(feature, ComparisonFeature):
            if feature.name not in self.comparison_features:
                self.comparison_features[feature.name] = {}
            self.comparison_features[feature.name][feature.models] = feature
        else:
            raise NotImplementedError(
                "Feature registration is only implemented for item-based, "
                + "model-based and comparative features!"
            )

    def get_feature(
        self,
        name: str,
        model: Optional[str] = None,
        models: Optional[Tuple[str, str]] = None,
    ) -> Feature:
        if name in self.item_features:
            return self.item_features[name]
        if name in self.model_features:
            if model is None:
                raise ValueError("Need to specify 'model' for model-based features!")
            return self.model_features[name].get(model, None)
        if name in self.comparison_features:
            relevant_features = self.comparison_features[name]

            if models is None:
                if len(relevant_features) > 1:
                    raise ValueError(
                        "Need to specify 'models' for comparative features which "
                        + "exist for several pairs of models!"
                    )
                if len(relevant_features) == 1:
                    return list(relevant_features.values())[0]
            else:
                sorted_models = tuple(sorted(list(models)))
                return self.comparison_features[name].get(sorted_models, None)

        return None

    def get_features(self, name: str) -> List[Feature]:
        features = []

        if name in self.item_features:
            features.append(self.item_features[name])
        if name in self.model_features:
            features.extend(self.model_features[name].values())
        if name in self.comparison_features:
            features.extend(self.comparison_features[name].values())

        return features

    def select(self, indices: List[int], name: str):
        """Create a new Experiment by sampling items with the given indices

        Keeps all features."""
        dataset = self.dataset.select(indices=indices)

        item_features = {
            n: self.item_features[n].select(indices=indices) for n in self.item_features
        }
        model_features = {
            n: {
                model: self.model_features[n][model].select(indices=indices)
                for model in self.model_features[n]
            }
            for n in self.model_features
        }
        comparison_features = {
            n: {
                models: self.comparison_features[n][models].select(indices=indices)
                for models in self.comparison_features[n]
            }
            for n in self.comparison_features
        }

        return self.__class__(
            name=name,
            dataset=dataset,
            item_features=item_features,
            model_features=model_features,
            comparison_features=comparison_features,
        )

    def compute_model_choices(self, model: Model) -> ModelChoices:
        """Compute choices and option log likelihoods for the given model.

        Features are registered so they can be re-used.
        If any of these features already exists, computation is skipped."""
        model_choices = self.get_feature("ModelChoices", model=model.name)
        if model_choices is None:
            lls = self.get_feature("OptionLogLikelihood", model=model.name)
            if lls is None:
                lls = OptionLogLikelihood.compute(self.dataset, model=model)
                self.register_feature(lls)

            model_choices = ModelChoices.compute(lls)
            self.register_feature(model_choices)
        return model_choices

    def compute_correctness(self, models: List[Model]):
        ground_truth = self.get_feature("GroundTruth")
        if ground_truth is None:
            raise PreconditionError(
                "PredictionCorrectness can only be computed if ground truth "
                + "information is available and added to the experiment!"
            )

        for model in models:
            model_choices = self.compute_model_choices(model=model)
            self.register_feature(
                PredictionCorrectness.compute(
                    model_choices=model_choices, ground_truth=ground_truth
                )
            )

    def compute_disagreement(self, models: Tuple[Model, Model]):
        assert len(models) == 2

        # Ensure that log likelihoods exist
        lls1 = self.get_feature("OptionLogLikelihood", model=models[0].name)
        if lls1 is None:
            lls1 = OptionLogLikelihood.compute(self.dataset, model=models[0])
            self.register_feature(lls1)
        lls2 = self.get_feature("OptionLogLikelihood", model=models[1].name)
        if lls2 is None:
            lls2 = OptionLogLikelihood.compute(self.dataset, model=models[1])
            self.register_feature(lls2)

        log_disagreement = LogDisagreement.compute(log_likelihoods=(lls1, lls2))
        self.register_feature(log_disagreement)
        self.register_feature(BinaryDisagreement.compute(log_disagreement))

    def compute_continuation_disagreement(self, models: Tuple[Model, Model], **kwargs):
        assert len(models) == 2

        tc1 = self.get_feature("TextContinuation", model=models[0].name)
        if tc1 is None:
            tc1 = TextContinuation.compute(
                dataset=self.dataset, model=models[0], **kwargs
            )
            self.register_feature(tc1)
        tc2 = self.get_feature("TextContinuation", model=models[1].name)
        if tc2 is None:
            tc2 = TextContinuation.compute(
                dataset=self.dataset, model=models[1], **kwargs
            )
            self.register_feature(tc2)

        cd = ContinuationDisagreement.compute(
            dataset=self.dataset, continuations=[tc1, tc2], models=models
        )
        self.register_feature(cd)

    def sample(
        self,
        num: Optional[int] = None,
        *,
        sampling_method: str = "random",
        ordering_scores: Optional[List[float]] = None,
        mask: Optional[List[bool]] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            scores_text = "scores" if ordering_scores is not None else "default"
            mask_text = ", masked" if mask is not None else ""
            name = (
                f"{self.name} Samples (num={num}, sampling={sampling_method}, "
                + f"order={scores_text}{mask_text})"
            )

        if mask is not None:
            assert len(mask) == len(self.dataset.items)

            if ordering_scores is not None:
                temp = np.argsort(ordering_scores)
                indices = np.array([ix for ix in temp if mask[ix]])
            else:
                indices = np.where(np.array(mask))[0]
        else:
            if ordering_scores is not None:
                indices = np.argsort(ordering_scores)
            else:
                indices = range(len(self.dataset.items))

        if num is None:
            num = len(indices)
        if sampling_method == "random":
            selected_indices = list(
                np.random.choice(indices, size=min(num, len(indices)), replace=False)
            )
        elif sampling_method == "first":
            selected_indices = indices[:num]
        elif sampling_method == "last":
            selected_indices = indices[::-1][:num]
        else:
            raise ValueError(
                "sampling_method must be one of ('random', 'first', 'last')"
            )

        return self.select(indices=selected_indices, name=name)

    def display_item(
        self,
        item_ix: int,
        feature_names: Optional[List[str]] = None,
    ):
        item = self.dataset.items[item_ix]
        print(f"ITEM ({item.id})")
        print(f'"""{item.prompt}"""')
        print(f"Options: {item.options}")

        if feature_names is None:
            feature_names = list(self.item_features.keys())
            feature_names.extend(list(self.model_features.keys()))
            feature_names.extend(list(self.comparison_features.keys()))

        print("\nFEATURES")
        for feature_name in feature_names:
            feature_list = self.get_features(feature_name)
            for feature in feature_list:
                if isinstance(feature, ItemFeature):
                    print(f"{feature_name}: {feature.values[item_ix]}")
                elif isinstance(feature, ModelFeature):
                    print(
                        f"{feature_name} - {feature.model}: {feature.values[item_ix]}"
                    )
                elif isinstance(feature, ComparisonFeature):
                    print(
                        f"{feature_name} - {feature.models}: {feature.values[item_ix]}"
                    )
                else:
                    print("Warning: Unknown feature type!")
                    print(f"{feature_name}: {feature.values[item_ix]}")

    def display_items(
        self,
        item_ixs: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if item_ixs is None:
            item_ixs = range(len(self.dataset.items))
        for ix in item_ixs:
            self.display_item(item_ix=ix, feature_names=feature_names)
            print("\n\n" + 20 * "-" + "\n\n")

    def feature_dependency_table(
        self,
        input_feature: Union[str, dict],
        output_feature: Union[str, dict] = "BinaryDisagreement",
        p_value: float = 0.95,
    ):
        """Create a table showing average values of `output_feature` for different
        values of `input_feature`.

        To select features, you can either pass the name as string, or a
        dictionary with keyword arguments (to be passed to `.get_feature()`).
        """
        if isinstance(input_feature, str):
            input_values = self.get_feature(input_feature).values
        else:
            input_values = self.get_feature(**input_feature).values

        data = []

        def make_data_element(input_value: str, output_values):
            if len(set(output_values)) < 2:
                conf_interval = None
            else:
                conf_interval = st.t.interval(
                    p_value,
                    len(output_values) - 1,
                    loc=np.mean(output_values),
                    scale=st.sem(output_values),
                )

            output_feature_vals = (
                f"{np.mean(output_values):.2f} "
                + f"({conf_interval[0]:.2f}-{conf_interval[1]:.2f})"
                if conf_interval is not None
                else f"{np.mean(output_values):.2f}"
            )

            d = {
                (
                    input_feature
                    if isinstance(input_feature, str)
                    else input_feature["name"]
                ): input_value,
                "#items": len(output_values),
                (
                    output_feature
                    if isinstance(output_feature, str)
                    else output_feature["name"]
                ): output_feature_vals,
            }
            return d

        for v in sorted(list(set(input_values))):
            mask = input_values == v
            value_selection = self.sample(mask=mask)

            if isinstance(output_feature, str):
                selection_values = value_selection.get_feature(output_feature).values
            else:
                selection_values = value_selection.get_feature(**output_feature).values

            data.append(
                make_data_element(input_value=v, output_values=selection_values)
            )

        if isinstance(output_feature, str):
            values = self.get_feature(output_feature).values
        else:
            values = self.get_feature(**output_feature).values

        data.append(make_data_element(input_value="OVERALL", output_values=values))

        return pd.DataFrame(data)

    def to_dict(self):
        d = {"name": self.name}
        d["dataset"] = self.dataset.to_dict()
        d["item_features"] = {
            name: self.item_features[name].to_dict() for name in self.item_features
        }
        d["model_features"] = {
            name: {
                model: self.model_features[name][model].to_dict()
                for model in self.model_features[name]
            }
            for name in self.model_features
        }
        d["comparison_features"] = {
            name: {
                models: self.comparison_features[name][models].to_dict()
                for models in self.comparison_features[name]
            }
            for name in self.comparison_features
        }
        return d

    @classmethod
    def from_dict(cls, d):
        args = {"name": d["name"], "dataset": Dataset.from_dict(d["dataset"])}
        args["item_features"] = {
            name: feature_from_dict(d["item_features"][name])
            for name in d["item_features"]
        }
        args["model_features"] = {
            name: {
                model: feature_from_dict(d["model_features"][name][model])
                for model in d["model_features"][name]
            }
            for name in d["model_features"]
        }
        args["comparison_features"] = {
            name: {
                models: feature_from_dict(d["comparison_features"][name][models])
                for models in d["comparison_features"][name]
            }
            for name in d["comparison_features"]
        }
        return cls(**args)

    def save(self, data_dir: Path):
        path = data_dir / Path(self.name + ".joblib")
        data = self.to_dict()
        dump(data, path)

    @classmethod
    def load(cls, data_dir: Path, name: str):
        path = data_dir / Path(name + ".joblib")
        return cls.from_dict(load(path))
