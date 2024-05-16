# Functions to load datasets for analysis
from typing import List, Tuple, Union
from pydantic import BaseModel


class Item(BaseModel):
    id: Union[int, str]
    prompt: str
    options: List[str]
    correct_index: int

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

