# Functions to load datasets for analysis
from typing import List, Tuple, Union, Any, Dict
from pydantic import BaseModel


class Item(BaseModel):
    id: Union[int, str]
    prompt: str
    options: List[str]


class Dataset(BaseModel):
    """Class to handle datasets"""
    name: str
    items: Tuple[Item, ...]
    features: Dict[str, Any] = {}  # entries of type ItemFeature

    def register_feature(self, feature):
        self.features[feature.name] = feature

    def get_feature(self,
            feature_name: str,
        ):
        return self.features.get(feature_name, None)

    def select(self, indices: List[int]):
        items = [self.items[i] for i in indices]
        features = {name: self.features[name].select(indices) for name in self.features}

        if not self.name.endswith(" selected"):
            name = self.name + " selected"
        else:
            name = self.name

        return self.__class__(name=name, items=tuple(items), features=features)

