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

    def select(self, indices: List[int]):
        items = [self.items[i] for i in indices]

        if not self.name.endswith(" selected"):
            name = self.name + " selected"
        else:
            name = self.name

        return self.__class__(name=name, items=tuple(items))

    def to_dict(self):
        d = {'name': self.name, 'items': [dict(item) for item in self.items]}
        return d

    @classmethod
    def from_dict(cls, d):
        assert 'name' in d and 'items' in d

        return cls(name=d['name'], items=[Item(**item) for item in d['items']])
