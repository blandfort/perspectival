# Perspectival

A Python-based toolkit for comparing transformers.
Perspectival goes beyond simple evaluation metrics like accuracy.
The goal rather is to identify in which situations models behave differently and to analyze why that is.

Notes:

* This repository is very early-stage! The main functionality should be working, but the number of implemented features is still limited.
* I'm looking for collaborators. If you're interested please contact me!


## Usage

TODO How to install and run

TODO Explain how to load models and set up experiments (linking also to the demo notebook)


## Status

### Urgent TODOs

- Document properly
  - Add how-to
  - Brush up demo notebook
- Adjust preprocessing / dataset loading to match reported performances on hellaswag (smallest apple model 5 points below, largest 10 points – but based on 100 samples from training set only)


### Current Status

* Tested on following environments: MacOS with Python3.11.0
* Supported models: Huggingface transformers (tested for Apple's OpenELM models, GPT2 and DistilGPT2)
* Supported datasets: Hellaswag, Rotten Tomatoes and part of Anthropic's advanced AI risk dataset (see `loader.py` for details)


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
