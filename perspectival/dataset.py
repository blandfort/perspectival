from typing import List, Tuple, Union
from pydantic import BaseModel


class Item(BaseModel):
    id: Union[int, str]
    prompt: str
    options: List[str]

    def get_statements(self) -> List[str]:
        """Get a list of statements corresponding to the different options.

        Log likelihoods should be computed for these statements."""
        statements = [
            (
                self.prompt
                + (
                    " "
                    if (not self.prompt.endswith(" ") and not option.startswith(" "))
                    else ""
                )
                + option
            ).strip()
            for option in self.options
        ]
        return statements

    def get_continuation_prompt(self) -> str:
        """Get a prompt that is used as input to generate continuations."""
        return self.prompt


class PlaceholderItem(Item):
    placeholder: str = "_"

    def get_statements(self) -> List[str]:
        """Get a list of statements corresponding to the different options.

        Log likelihoods should be computed for these statements."""
        statements = []
        for option in self.options:
            statement = self.prompt.replace(self.placeholder, option)

            # Remove extra whitespaces
            statement = statement.strip()

            statements.append(statement)
        return statements

    def get_continuation_prompt(self) -> str:
        """Get a prompt that is used as input to generate continuations."""
        raise ValueError("Continuation doesn't work for placeholder items!")


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
        d = {"name": self.name, "items": [dict(item) for item in self.items]}
        return d

    @classmethod
    def from_dict(cls, d):
        assert "name" in d and "items" in d

        return cls(name=d["name"], items=[Item(**item) for item in d["items"]])
