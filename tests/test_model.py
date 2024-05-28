import pytest
import numpy as np

from perspectival.dataset import Item
from perspectival.model import Transformer, SimpleTransformer


@pytest.fixture
def transformer_model():
    #TODO problem in cicd pipeline will be that this model needs to be downloaded...
    model = Transformer('gpt2', lazy_loading=False)
    return model

@pytest.fixture
def simple_transformer_model():
    #TODO problem in cicd pipeline will be that this model needs to be downloaded...
    model = SimpleTransformer('gpt2', lazy_loading=False)
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
    print("Log likelihoods:", log_likelihoods)
    assert len(log_likelihoods) == len(item.options)
    assert np.max(log_likelihoods) < 0
    assert item.options[np.argmin(log_likelihoods)] == item.options[unlikely_index]

def test_model_equivalence(transformer_model, simple_transformer_model, test_data):
    item, unlikely_index = test_data
    log_likelihoods = transformer_model.compute_option_log_likelihoods(items=[item])[0]
    simple_log_likelihoods = simple_transformer_model.compute_option_log_likelihoods(items=[item])[0]
    print("Log likelihoods:", log_likelihoods, simple_log_likelihoods)

    assert len(log_likelihoods) == len(simple_log_likelihoods)
    assert np.max(simple_log_likelihoods) < 0

    ranks = np.array(log_likelihoods).argsort().argsort()
    simple_ranks = np.array(simple_log_likelihoods).argsort().argsort()
    assert np.all(ranks == simple_ranks)

#TODO! For compute_token_log_likelihood we want to make sure that adding more text leads to lower overall value
#TODO! Do the same for apple's model!
#TODO Also test for left-padding (e.g. with apple's model checking if both tokenizers give the same values)
