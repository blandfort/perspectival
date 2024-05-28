import pytest
import numpy as np

from perspectival.dataset import Item
from perspectival.model import Transformer, SimpleTransformer


@pytest.fixture
def transformer_model():
    model = Transformer('gpt2', lazy_loading=False)
    return model

# Define a parametrized fixture that includes all your test cases
@pytest.fixture(params=[
    (Item(id=0, prompt="This is ", options=["a", "the", "my", "banana four"]), -1),
    (Item(id=1, prompt="Welcome to ", options=["school", "asdf", "the world"]), 1),
    # Add more test cases here as tuples of (Item, unlikely_index)
])
def test_data(request):
    return request.param

def test_obvious_choices(transformer_model, test_data):
    item, unlikely_index = test_data
    log_likelihoods = transformer_model.compute_option_log_likelihoods(items=[item])[0]
    assert len(log_likelihoods) == len(item.options)
    assert item.options[np.argmin(log_likelihoods)] == item.options[unlikely_index]

