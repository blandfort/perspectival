import pytest

from perspectival.model import Transformer


# Parametrize the fixture to create different Transformer models
@pytest.fixture(params=[
    ('gpt2', False),
    ('apple/OpenELM-270M', False),
])
def transformer_model(request):
    #TODO problem in cicd pipeline will be that this model needs to be downloaded...
    model_name, lazy_loading = request.param
    return Transformer(model_name, lazy_loading=lazy_loading, model_kwargs={'trust_remote_code': True})

