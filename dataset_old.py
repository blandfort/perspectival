# Functions to load datasets for analysis
from typing import List, Union, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from datasets import load_dataset
import numpy as np


class Item(BaseModel):
    id: Union[int, str]
    prompt: str
    options: List[str]
    correct_index: int

#TODO Might wanna switch away from BaseModel,
# to make sure that subset() actually ends up having numpy arrays
class Dataset(BaseModel):
    """Class to handle datasets

    Note that we assume items to be added initially and not changed.
    Features have to be aligned with items so that for any feature 'X',
    `features[X][i]` contains the corresponding feature for `items[i]`."""
    name: str
    items: Tuple[Item, ...]
    features: Dict[str, Tuple[Any, ...]] = {}
    predictions: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
    comparisons: Dict[Tuple[str, str], Dict[str, Tuple[Any, ...]]] = {}

    def set_features(self, name: str, values: List[Union[int, float, str]]):
        self.features[name] = np.array(values)

    def get_features(self, name: str) -> np.array:
        return self.features[name]

    def has_features(self, name: str) -> bool:
        return (name in self.features)

    def set_predictions(self,
            model_name: str,
            predictions_name: str,
            values: List[Union[int, float, str]]
        ):
        if model_name not in self.predictions:
            self.predictions[model_name] = {}
        self.predictions[model_name][predictions_name] = np.array(values)

    def get_predictions(self,
            model_name: Optional[str]=None,
            predictions_name: Optional[str]=None
        ) -> Union[np.array, Dict[str, np.array]]:
        if model_name is not None and predictions_name is not None:
            return self.predictions[model_name][predictions_name]
        elif predictions_name is not None:
            predictions = {}
            for model_name in self.predictions:
                if predictions_name in self.predictions[model_name]:
                    predictions[model_name] = self.predictions[model_name][predictions_name]
            return predictions
        elif model_name is not None:
            return self.predictions[model_name]
        else:
            raise ValueError("Need to specify model_name or predictions_name!")

    def has_predictions(self, model_name: str, predictions_name: str) -> bool:
        if model_name in self.predictions and predictions_name in self.predictions[model_name]:
            return True
        else:
            return False

    def set_comparison(self, model_names: Tuple[str, str], comparison_name: str, values: List[Union[int, float, str]]):
        key = tuple(sorted(model_names))
        if key not in self.comparisons:
            self.comparisons[key] = {}
        self.comparisons[key][comparison_name] = np.array(values)

    def get_comparison(self,
            model_names: Optional[Tuple[str, str]]=None,
            comparison_name: Optional[str]=None,
        ) -> Union[np.array, Dict[str, np.array]]:
        if model_names is not None and comparison_name is not None:
            key = tuple(sorted(model_names))
            return self.comparisons[key][comparison_name]
        elif comparison_name is not None:
            comparison = {}
            for key in self.comparisons:
                if comparison_name in self.comparisons[key]:
                    comparison[key] = self.comparisons[key][comparison_name]
            return comparison
        elif model_names is not None:
            key = tuple(sorted(model_names))
            return self.comparisons[key]
        else:
            raise ValueError("Need to specify model_names or comparison_name!")

    def has_comparisons(self, model_names: Tuple[str, str], comparison_name: str) -> bool:
        key = tuple(sorted(model_names))
        if key in self.comparisons and comparison_name in self.comparisons[key]:
            return True
        else:
            return False

    def subset(self, indices: List[int], name: str):
        """Create a new dataset by sampling items with the given indices

        Keeps features, predictions and comparisons"""
        items = [self.items[ix] for ix in indices]

        features = {name: np.array([self.features[name][ix] for ix in indices])
            for name in self.features}
        predictions = {}
        for model_name in self.predictions:
            predictions[model_name] = {name: np.array([self.predictions[model_name][name][ix] for ix in indices])
                for name in self.predictions[model_name]}
        comparisons = {}
        for key in self.comparisons:
            comparisons[key] = {name: np.array([self.comparisons[key][name][ix] for ix in indices])
                for name in self.comparisons[key]}

        return Dataset(
            name=name,
            items=items,
            features=features,
            predictions=predictions,
            comparisons=comparisons,
        )


def load_rotten_tomatoes(shuffle: bool = True, split: str = 'train') -> Dataset:
    dataset_name = "rotten_tomatoes"

    # Load the dataset
    raw_dataset = load_dataset(dataset_name, split=split)

    items = []

    # Define options for sentiment based on the labels used in the dataset
    options = ['negative', 'positive']  # Define options for sentiment

    for ix, element in enumerate(raw_dataset):
        prompt = f"Q: What is the sentiment of the text below?\n\n'{element['text']}'\n\nA: The sentiment is"
        correct_index = element['label']  # Label directly corresponds to the index in options
        item_id = f"{split}_{ix}"
        items.append(Item(id=item_id, prompt=prompt, options=options, correct_index=correct_index))

    if shuffle:
        np.random.shuffle(items)  # Shuffle the list of items to randomize the order

    # Create and return a Dataset instance
    dataset = Dataset(name=dataset_name, items=items)
    return dataset


def sample_dataset(
        dataset: Dataset,
        num: Optional[int]=None,
        sampling_method: str='first',
        ordering_scores: Optional[List[float]]=None,
        mask: Optional[List[bool]]=None,
        name: Optional[str]=None,
    ) -> Dataset:
    if name is None:
        scores_text = "scores" if ordering_scores is not None else "default"
        mask_text = ", masked" if mask is not None else ""
        name = f'{dataset.name} Samples (num={num}, sampling={sampling_method}, order={scores_text}{mask_text})'

    if mask is not None:
        assert len(mask)==len(dataset.items)

        indices = np.where(np.array(mask))[0]
        if ordering_scores is not None:
            indices = np.argsort(np.array(ordering_scores)[indices])
    else:
        if ordering_scores is not None:
            indices = np.argsort(ordering_scores)
        else:
            indices = range(len(dataset.items))

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

    return dataset.subset(indices=selected_indices, name=name)

