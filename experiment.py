from typing import List, Union, Dict, Tuple, Optional, Type
import numpy as np
from joblib import dump, load
from pathlib import Path

from interfaces import Feature, ItemFeature, ModelFeature, ComparisonFeature
from dataset import Dataset


class Experiment:
    name: str
    dataset: Dataset
    item_features: Dict[str, ItemFeature] = {}
    model_features: Dict[str, Dict[str, ModelFeature]] = {}
    comparison_features: Dict[str, Dict[Tuple[str, str], ComparisonFeature]] = {}

    def __init__(self,
            name,
            dataset,
            item_features = None,
            model_features = None,
            comparison_features = None,
            features: Optional[List[Feature]] = None,
        ):
        self.name = name
        self.dataset = dataset

        self.item_features = item_features if item_features is not None else {}
        self.model_features = model_features if model_features is not None else {}
        self.comparison_features = comparison_features if comparison_features is not None else {}

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
            raise NotImplementedError("Feature registration is only implemented for item-based, model-based and comparative features!")

    def get_feature(self,
            feature_name: str,
            model: Optional[str]=None,
            models: Optional[Tuple[str, str]]=None,
        ) -> Feature:
        if feature_name in self.item_features:
            return self.item_features[feature_name]
        elif feature_name in self.model_features:
            if model is None:
                raise ValueError("Need to specify 'model' for model-based features!")
            if model in self.model_features[feature_name]:
                return self.model_features[feature_name][model]
            else:
                return None
        elif feature_name in self.comparison_features:
            if models is None:
                raise ValueError("Need to specify 'models' for comparative features!")
            sorted_models = tuple(sorted(list(models)))
            if sorted_models in self.comparison_features[feature_name]:
                return self.comparison_features[feature_name][sorted_models]
            else:
                return None
        else:
            return None

    def get_features(self, feature_name: str) -> List[Feature]:
        features = []

        if feature_name in self.item_features:
            features.append(self.item_features[feature_name])
        if feature_name in self.model_features:
            features.extend(self.model_features[feature_name].values())
        if feature_name in self.comparison_features:
            features.extend(self.comparison_features[feature_name].values())

        return features

    def select(self, indices: List[int], name: str):
        """Create a new Experiment by sampling items with the given indices

        Keeps all features."""
        dataset = self.dataset.select(indices=indices)

        item_features = {n: self.item_features[n].select(indices=indices)
                        for n in self.item_features}
        model_features = {n: {model: self.model_features[n][model].select(indices=indices)
                            for model in self.model_features[n]}
                        for n in self.model_features}
        comparison_features = {n: {models: self.comparison_features[n][models].select(indices=indices)
                            for models in self.comparison_features[n]}
                        for n in self.comparison_features}

        return self.__class__(
            name=name,
            dataset=dataset,
            item_features=item_features,
            model_features=model_features,
            comparison_features=comparison_features,
        )

    def sample(
            self,
            num: Optional[int]=None,
            sampling_method: str='random',
            ordering_scores: Optional[List[float]]=None,
            mask: Optional[List[bool]]=None,
            name: Optional[str]=None,
        ):
        if name is None:
            scores_text = "scores" if ordering_scores is not None else "default"
            mask_text = ", masked" if mask is not None else ""
            name = f'{self.name} Samples (num={num}, sampling={sampling_method}, order={scores_text}{mask_text})'

        if mask is not None:
            assert len(mask)==len(self.dataset.items)

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
        if sampling_method=='random':
            selected_indices = list(np.random.choice(indices, size=min(num, len(indices)), replace=False))
        elif sampling_method=='first':
            selected_indices = indices[:num]
        elif sampling_method=='last':
            selected_indices = indices[::-1][:num]
        else:
            raise ValueError("sampling_method must be one of ('random', 'first', 'last')")

        return self.select(indices=selected_indices, name=name)

    def save(self, data_dir: Path):
        path = data_dir / Path(self.name + '.joblib')
        dump(self, path)

    @classmethod
    def load(cls, data_dir: Path, name: str):
        path = data_dir / Path(name + '.joblib')
        return load(path)

    def display_item(
            self,
            item_ix: int,
            feature_names: Optional[List[str]]=None,
        ):
        item = self.dataset.items[item_ix]
        print(f"ITEM ({item.id})")
        print(f'"""{item.prompt}"""')
        print(f"Options: {item.options}")
        print(f"Ground truth: {item.options[item.correct_index]}")

        if feature_names is None:
            feature_names = list(self.item_features.keys())
            feature_names.extend(list(self.model_features.keys()))
            feature_names.extend(list(self.comparison_features.keys()))

        print("\nFEATURES")
        for feature_name in feature_names:
            feature_list = self.get_features(feature_name)
            for feature in feature_list:
                if isinstance(feature, ItemFeature):
                    print(feature_name, feature.values[item_ix])
                elif isinstance(feature, ModelFeature):
                    print(feature_name, feature.model, feature.values[item_ix])
                elif isinstance(feature, ComparisonFeature):
                    print(feature_name, feature.models, feature.values[item_ix])
                else:
                    print("Warning: Unknown feature type!")
                    print(feature_name, feature.values[item_ix])

    def display_items(
            self,
            item_ixs: Optional[List[int]]=None,
            feature_names: Optional[List[str]]=None,
        ):
        if item_ixs is None:
            item_ixs = range(len(self.dataset.items))
        for ix in item_ixs:
            self.display_item(item_ix=ix, feature_names=feature_names)
            print('\n\n' + 20*'-' + '\n\n')
