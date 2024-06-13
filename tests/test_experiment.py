import os
import pytest

from perspectival.loader import load_hellaswag
from perspectival.experiment import Experiment


TEST_DATA_DIR = "tests/test_data"


@pytest.fixture
def experiment():
    dataset, features = load_hellaswag()
    exp = Experiment(dataset=dataset, name="HellaSwag_TEST", features=features)
    return exp


def test_experiment_dictionary_conversion(experiment):
    # TODO add features from two models as well!

    d = experiment.to_dict()
    restored_experiment = Experiment.from_dict(d)

    # Verify that all entries are still there
    # TODO Ideally we implement '==' for Experiment
    assert experiment.name == restored_experiment.name
    for name in experiment.item_features:
        assert name in restored_experiment.item_features
    for name in experiment.model_features:
        assert name in restored_experiment.model_features
        for model in experiment.model_features[name]:
            assert model in restored_experiment.model_features[name]
    for name in experiment.comparison_features:
        assert name in restored_experiment.comparison_features
        for models in experiment.comparison_features[name]:
            assert models in restored_experiment.comparison_features[name]


def test_experiment_saving_and_loading(experiment):
    # TODO add features from two models as well!

    if not os.path.isdir(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)

    experiment.save(TEST_DATA_DIR)
    restored_experiment = Experiment.load(TEST_DATA_DIR, experiment.name)

    # Verify that all entries are still there
    # TODO Ideally we implement '==' for Experiment
    assert experiment.name == restored_experiment.name
    for name in experiment.item_features:
        assert name in restored_experiment.item_features
    for name in experiment.model_features:
        assert name in restored_experiment.model_features
        for model in experiment.model_features[name]:
            assert model in restored_experiment.model_features[name]
    for name in experiment.comparison_features:
        assert name in restored_experiment.comparison_features
        for models in experiment.comparison_features[name]:
            assert models in restored_experiment.comparison_features[name]
