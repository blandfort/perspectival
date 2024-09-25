import pytest
import numpy as np

from perspectival.dataset import Item
from perspectival.model import compute_token_log_likelihood


# Define a parametrized fixture that includes all your test cases
@pytest.fixture(
    params=[
        (Item(id=0, prompt="This is ", options=["a", "the", "my", "banana four"]), -1),
        (Item(id=1, prompt="Welcome to ", options=["school", "asdf", "the world"]), 1),
        (
            Item(
                id=2, prompt="When you test,", options=[" be sure to", " you", ". No"]
            ),
            2,
        ),
        (Item(id=3, prompt="", options=["One of the blanket", "The sky is blue."]), 0),
        (
            Item(
                id=4, prompt="", options=["Hello world!", "Bello world!", "Hello girl!"]
            ),
            1,
        ),
        (Item(id=5, prompt="", options=["I am", "I go", "The am"]), 2),
        # Add more test cases here as tuples of (Item, unlikely_index)
    ]
)
def test_data(request):
    return request.param


def test_obvious_choices(transformer_model, test_data):
    item, unlikely_index = test_data
    log_likelihoods = transformer_model.compute_option_log_likelihoods(items=[item])[0]
    print("Log likelihoods:", log_likelihoods)
    assert len(log_likelihoods) == len(item.options)
    assert np.max(log_likelihoods) < 0
    assert item.options[np.argmin(log_likelihoods)] == item.options[unlikely_index]


def test_option_ranking(transformer_model, test_data):
    item, _ = test_data
    log_likelihoods = transformer_model.compute_option_log_likelihoods(items=[item])[0]

    simple_log_likelihoods = []
    for option in item.options:
        text = item.prompt + option
        log_ls = np.sum(
            compute_token_log_likelihood(
                transformer_model.model, transformer_model.tokenizer, text=text
            )["log_likelihoods"]
        )
        simple_log_likelihoods.append(log_ls)
    print("Log likelihoods:", log_likelihoods, simple_log_likelihoods)

    assert len(log_likelihoods) == len(simple_log_likelihoods)
    assert np.max(simple_log_likelihoods) < 0

    ranks = np.array(log_likelihoods).argsort().argsort()
    simple_ranks = np.array(simple_log_likelihoods).argsort().argsort()
    assert np.all(ranks == simple_ranks)


@pytest.mark.parametrize(
    "text, extension",
    [
        ("This is", " a"),
        ("Let us take a", " moment"),
    ],
)
def test_compute_token_log_likelihood(transformer_model, text, extension):
    """We want to make sure that adding more text leads to lower overall value
    (unless tokens are merged)"""
    transformer = transformer_model.model
    tokenizer = transformer_model.tokenizer

    log_ls = compute_token_log_likelihood(transformer, tokenizer, text)[
        "log_likelihoods"
    ]
    ext_log_ls = compute_token_log_likelihood(transformer, tokenizer, text + extension)[
        "log_likelihoods"
    ]

    assert sum(log_ls) > sum(ext_log_ls)


# TODO Also test for left-padding (e.g. with apple's model checking
# if both tokenizers give the same values)
