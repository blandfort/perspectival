import pytest
import numpy as np

from perspectival.features import LogDisagreement, OptionLogLikelihood


@pytest.mark.parametrize("item_log_likelihoods1, item_log_likelihoods2, min_value, max_value", [
    ((-100, -0.01), (-0.01, -100), .97, 1),  # Perfect disagreement
    ((-100, -110), (-110, -100), .05, .2),
    ((-1, -10), (-3, -4), 0, 0),  # Same choice
    # Additional irrelevant options shouldn't have an effect
    ((-100, -0.01, -500), (-0.01, -100, -500), .97, 1),
    ((-500, -100, -1000, -110), (-500, -110, -1000, -100), .05, .2),
])
def test_disagreement_score(item_log_likelihoods1, item_log_likelihoods2, min_value, max_value):
    model1 = 'dummy1'
    model2 = 'dummy2'
    log_likelihoods1 = OptionLogLikelihood(model=model1, values=[item_log_likelihoods1])
    log_likelihoods2 = OptionLogLikelihood(model=model2, values=[item_log_likelihoods2])

    print(log_likelihoods1.to_dict())
    disagreements = LogDisagreement.compute(log_likelihoods=(log_likelihoods1, log_likelihoods2))
    assert disagreements.models==('dummy1', 'dummy2')

    print(disagreements.to_dict())
    disagreement_score = disagreements.values[0]
    assert max_value >= disagreement_score >= min_value
