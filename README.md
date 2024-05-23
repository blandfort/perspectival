# Perspectival

A Python-based toolkit for comparing transformers.

## TODO

- Document properly
  - Supported models: currently working with some Huggingface transformers (list ones I tried)
  - Supported datasets: hellaswag, rotten tomatoes, anthropic
  - Add how-to
  - Brush up demo notebook
- Run some meaningful analysis and write about it
- Adjust preprocessing / dataset loading to match reported performances on hellaswag (smallest apple model 5 points below, largest 10 points – but based on 100 samples from training set only)

### Backlog

- Feature dependencies: Specify a primary feature (comparison one) and another feature and visualize the dependency (E.g. how is disagreement different for categories; or does the presence of some regex pattern correlate with one model being better?)
  - Offer different types of visualization/analyses depending also on the data type of feature (bool, discrete, float)
- In-depth analysis of examples, also looking at token-based stuff and extending to architecture-specific comparisons like differences in attention
- Adding explanations and evaluate how much of a comparison feature one or several features can explain
- Add analyses based on generation
- Compare more than two models (for quite later): Mostly requires computing different comparison features

Some more ideas for later:

- For examples, also make it possible to view most similar ones (ideally based on embeddings), to quickly check some intuitions like "Additional structure in the prompt like [header] leads to more disagreement"
