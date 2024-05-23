# Perspectival

A Python-based toolkit for comparing transformers.
Perspectival goes beyond simple evaluation metrics like accuracy.
The goal rather is to identify in which situations models behave differently and to analyze why that is.

Notes:

* This repository is very early-stage! The main functionality should be working, but the number of implemented features is still limited.
* I'm looking for collaborators. If you're interested please contact me!


## Usage

### Installation

TODO How to install and run (install requirements, install the repo)


### Workflow

1. Initialize two models (subclasses of Model)
2. Load a dataset
3. Set up an Experiment
4. Add features to the Experiment by running comparisons
5. Explore the results

You can find full examples in [demo.ipynb](demo.ipynb).

TODO Below, for each step specify the interface as input and output, then give some examples and details


### Initializing Models

Format: Models used for experiments have to implement the interface Model (see [interfaces.py](interfaces.py)).

Available models:

- Model classes from [model.py](model.py)
  - Currently, this includes two classes to use Huggingface transformers (tested for Apple's OpenELM models, GPT2 and DistilGPT2)
- You can implement your own model class (as subclass of Model)


### Loading Datasets

Format: Datasets used for experiments should be subclasses of Dataset. Essentially, a Dataset has a name and list of Items, where each Item again has a certain format. If you want to use item-based features which are not part of the Item interface (e.g. categories), you can use ItemFeature. (See [interfaces.py](interfaces.py) for details.)

Available datasets:

- Datasets (and features) returned by functions in [loader.py](loader.py).
  - Currently, this includes Hellaswag, Rotten Tomatoes and part of Anthropic's advanced AI risk dataset (see [loader.py](loader.py) for details)
- You can implement own dataset classes (as subclass of Dataset)


### Setting up an Experiment

TODO

### Adding Features to Experiments

TODO

### Exploring Results

TODO



## Status

### Urgent TODOs

- Document properly
  - Add how-to
  - Brush up demo notebook
- Adjust preprocessing / dataset loading to match reported performances on hellaswag (smallest apple model 5 points below, largest 10 points – but based on 100 samples from training set only)


### Plans

The following extensions are planned soon, most likely in this order:

- Feature dependencies: Specify a primary feature (comparison one) and another feature and visualize the dependency (E.g. how is disagreement different for categories; or does the presence of some regex pattern correlate with one model being better?)
  - Offer different types of visualization/analyses depending also on the data type of feature (bool, discrete, float)
- In-depth analysis of examples, also looking at token-based features and extending to architecture-specific comparisons like differences in attention
- Adding explanations and evaluate how much of a comparison feature one or several features can explain

Some more ideas for later:

- Add analyses based on generation
- For examples, also make it possible to view most similar ones (ideally based on embeddings), to quickly check some intuitions like "Additional structure in the prompt like [header] leads to more disagreement"
- Compare more than two models: Mostly requires computing different comparison features
