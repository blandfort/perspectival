# Perspectival

A Python-based toolkit for comparing transformers.
Perspectival goes beyond simple evaluation metrics like accuracy.
The goal rather is to identify in which situations models behave differently and to analyze why that is.

Notes:

* This repository is very early-stage! The main functionality should be working, but the number of implemented features is still limited.
* I'm looking for collaborators. If you're interested please contact me!


## Usage

### Installation

1. Download the repository using `git clone`
2. cd to the repository
3. Optional: Set up and activate a virtual environment
4. Install Perspectival with `pip install -e .`

(So far tested on MacOS with Python3.11.0.)


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

- Model classes from [model.py](perspectival/model.py)
  - Currently, this includes two classes to use Huggingface transformers (tested for Apple's OpenELM models, GPT2 and DistilGPT2)
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

Format: Features used for experiments have to implement the interface Feature (see [interfaces.py](perspectival/interfaces.py). There are three types of features to use:

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


## Status

### Urgent TODOs

- Adjust preprocessing / dataset loading to match reported performances on hellaswag (smallest apple model 5 points below, largest 10 points – but based on 100 samples from training set only)


### Plans

The following extensions are planned soon, most likely in this order:

- Feature dependencies: Specify a primary feature (comparison one) and another feature and visualize the dependency (E.g. how is disagreement different for categories; or does the presence of some regex pattern correlate with one model being better?)
  - Offer different types of visualization/analyses depending also on the data type of feature (bool, discrete, float)
- In-depth analysis of examples, also looking at token-based features and extending to architecture-specific comparisons like differences in attention
- Adding explanations and evaluate how much of a comparison feature one or several features can explain

Usability features to add:

- Show progress when computing log likelihoods
- Make saving and loading experiments more robust (changes to Experiment class can lead loading of previous experiments to fail)

Some more ideas for later:

- Add analyses based on generation
- For examples, also make it possible to view most similar ones (ideally based on embeddings), to quickly check some intuitions like "Additional structure in the prompt like [header] leads to more disagreement"
- Compare more than two models: Mostly requires computing different comparison features
