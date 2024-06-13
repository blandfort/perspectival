# Perspectival

A Python-based toolkit for comparing transformers.
Perspectival goes beyond simple evaluation metrics like accuracy.
The goal rather is to identify in which situations models behave differently and to analyze why that is.

Notes:

* This repository is early-stage! (The main functionality should be working though.)
* I'm looking for collaborators. If you're interested please contact me!


## Usage

### Installation

To install the `perspectival` package, run:

```bash
git clone https://github.com/blandfort/perspectival
cd perspectival
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

(So far tested on MacOS with Python3.11.0 and Linux with Python3.10.)


### Workflow

1. Initialize two models (subclasses of Model)
2. Load a dataset
3. Set up an Experiment
4. Add features to the Experiment by running comparisons
5. Explore the results

You can find full examples in [examples/](examples/).


### Initializing Models

Format: Models used for experiments have to implement the interface Model (see [interfaces.py](perspectival/interfaces.py)).

Available models:

- Model classes from [model.py](perspectival/model.py), the standard class being `Transformer` to use HuggingFace transformers
  - Tested for several models including Apple's OpenELM models, GPT2, DistilGPT2, MPT-7B, Qwen2
  - You can pass model arguments when initializing, e.g. for authenticating with a HuggingFace token, or to use half-precision
  - Note: For some models you need to install additional packages (such as accelerate, protobuf, sentencepiece, einops)
- You can implement your own model class (as subclass of Model)


### Loading Datasets

Format: Datasets used for experiments should be subclasses of Dataset. Essentially, a Dataset has a name and list of Items, where each Item again has a certain format. If you want to use item-based features which are not part of the Item interface (e.g. categories), you can use ItemFeature. (See [interfaces.py](perspectival/interfaces.py) for details.)

Available datasets:

- Datasets (and features) returned by functions in [loader.py](perspectival/loader.py).
  - Currently, this includes Hellaswag, Rotten Tomatoes and part of Anthropic's advanced AI risk dataset (see [loader.py](perspectival/loader.py) for details)
- You can implement own dataset classes (as subclass of Dataset)


### Setting up an Experiment

To run experiments, use Experiment from [experiment.py](perspectival/experiment.py). An Experiment is initialized by passing a Dataset, a name and optionally a list of Feature.

You can also work with a subset of the dataset by calling the `experiment.sample` method.
For example, this allows you to run brief sanity checks and verifying that everything is working before starting longer-running computations.


### Adding Features to Experiments

Format: Features used for experiments have to implement the interface Feature (see [interfaces.py](perspectival/interfaces.py)). There are three types of features to use:

- ItemFeature: These features don't depend on any model but either come with the dataset or are computed without the models to be analyzed (e.g. based on regular expressions on the item text)
- ModelFeature: These features are computed for a single model. An example is OptionLogLikelihood, which computes the log likelihoods a model assigns to the different options described in an item.
- ComparisonFeature: These features are based on a pair of models. For example, LogDisagreement measures the amount of disagreement for each item.

Usage:

- Create a new feature either by using its init function and passing the values directly, or calling its compute method.
- Add features to the experiment by calling `experiment.register_feature(feature)`

Available features:

- In [features.py](perspectival/features.py) you can find several feature classes
- You can also implement own feature classes as subclass of one of the three classes ItemFeature, ModelFeature or ComparisonFeature


### Exploring Results

Once you have an Experiment with interesting features, you can

* Save the experiment with `experiment.save`
* Use the `experiment.sample` method to look at interesting cases (e.g. items with highest disagreement scores)
* Display items of choice together with features, using `experiment.display_items`
* Look at token-level differences, using `perspectival.inspect` (for an example, see [token_level_differences.ipynb](examples/token_level_differences.ipynb))


## Notes on Cloud Usage

If you want to run experiments in the cloud (which typically makes sense if you don't have a local GPU), the easiest way would be to rent a VM.

So far, I successfully ran experiments on a pod from RunPod. Some notes you might find helpful to do the same:

- Pod: 24GB of GPU memory, 50GB disk and 20GB pod volume, using PyTorch image
  - This was for running models such as Mistral-7B-v0.3 in half precision; for full precision or larger models you will need bigger GPUs
  - This cost <0.50$/h while running the pod, and <10$/month when the pod was paused (but still existing)
  - If you want to access models from gated repos, define a RunPod secret and pass it to the pod as environment variable (under "Edit Template" when setting up)
- How to set up perspectival on a pod:
  - Open jupyterlab on the pod
  - In jupyterlab, start a terminal and install the package
    - Skip the creation of the virtual environment, as PyTorch is already installed globally (and otherwise you need to manually add a new jupyter kernel)
    - Should be inside the `/workspace` directory if you want it to persist after stopping the pod!)
  - In case you want HuggingFace models to persist after pausing the pod, set the environment variable `HF_HOME` to a path within `/workspace`


## Development

To make sure that your changes comply to best practices, set up pre-commit hooks:

- Install extra requirements (see [setup.py](setup.py))
- Install pre-commit hooks: `pre-commit install`


## Plans

I am considering to extend the package with architecture-specific comparisons such as checking differences in attention. For example, this could help to look deeper into the differences between base models and their fine-tuned versions.
