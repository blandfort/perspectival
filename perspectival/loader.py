import re
import json
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Optional, Tuple, List

from .features import Category, GroundTruth
from .dataset import Dataset, Item
from .interfaces import Feature



def load_hellaswag(split: str = 'train') -> Tuple[Dataset, List[Feature]]:
    """Load the HellaSwag dataset.

    Also see https://huggingface.co/datasets/Rowan/hellaswag"""
    dataset_name = 'hellaswag'

    # Load the dataset
    raw_dataset = load_dataset(dataset_name, split=split)

    items = []
    ground_truth = []

    for ix, element in enumerate(raw_dataset):
        prompt = element['ctx']
        options = element['endings']

        # Label directly corresponds to the index in options
        correct_index = int(element['label']) if split!='test' else None
        # test split doesn't have labels

        item_id = f"{split}_{ix}"
        items.append(Item(id=item_id, prompt=prompt, options=options))
        ground_truth.append(correct_index)

    # Create and return a Dataset instance
    dataset = Dataset(name=dataset_name, items=items)
    if split!='test':
        features = [GroundTruth(values=ground_truth)]
    else:
        features = []
    return dataset, features

def load_rotten_tomatoes(split: str = 'train') -> Tuple[Dataset, List[Feature]]:
    dataset_name = "rotten_tomatoes"

    # Load the dataset
    raw_dataset = load_dataset(dataset_name, split=split)

    items = []
    ground_truth = []

    # Define options for sentiment based on the labels used in the dataset
    options = ['negative', 'positive']  # Define options for sentiment

    for ix, element in enumerate(raw_dataset):
        prompt = f"Question: What is the sentiment of the text below?\n\nText: '{element['text']}'\n\nAnswer: The sentiment is"
        correct_index = element['label']  # Label directly corresponds to the index in options
        item_id = f"{split}_{ix}"
        items.append(Item(id=item_id, prompt=prompt, options=options))
        ground_truth.append(correct_index)

    # Create and return a Dataset instance
    dataset = Dataset(name=dataset_name, items=items)
    features = [GroundTruth(values=ground_truth)]
    return dataset, features


def parse_anthropic_item(text):
    # Extract the question if present, otherwise use a default message
    question_match = re.search(r"(Question: )?(.*?)(?=\nChoices:)", text, re.DOTALL)
    question = question_match.group(2) #if question_match else "No question provided."

    # Extract the choices and remove the leading identifiers like (A), (B), etc.
    choices = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\n|\Z)", text)
    choice_list = [(choice[0], choice[1]) for choice in choices]
    choice_letters = [choice[0] for choice in choice_list]
    assert choice_letters==sorted(choice_letters)

    return question, [choice[1] for choice in choice_list]

def load_anthropic_eval_data(repo_path: Path, dataset: str,
        subdir: Optional[str]=None,
    ) -> Tuple[Dataset, List[Feature]]:
    """Load the advanced AI risk dataset from Anthropic

    The dataset needs to be downloaded from https://github.com/anthropics/evals
    first.

    Note that we use 'correct_index' to refer to options that don't match the
    corresponding behavior."""
    if dataset=='advanced-ai-risk':
        assert subdir in ['human_generated_evals', 'lm_generated_evals', 'prompts_for_few_shot_generation']

        items = []
        category_features = []

        dir_path = repo_path / Path(dataset) / Path(subdir)
        for path in os.listdir(dir_path):

            if path.endswith('.jsonl'):
                category = path.rsplit('.', 1)[0]
                raw_items = [json.loads(line) for line in open(dir_path / Path(path), 'r')]
                #print(category, len(raw_items))

                for ix, raw_item in enumerate(raw_items):
                    item_id = f"{dataset}/{subdir}/{category}/{ix}"

                    # Old version
                    #if raw_item['question'].lower().startswith('question'):
                    #    prompt = raw_item['question']
                    #else:
                    #    prompt = f"Question: {raw_item['question']}\n\nAnswer:"
                    #options = ["(A)", "(B)"]  # Note that we add a whitespace in processing
                    #correct_index = options.index(raw_item['answer_not_matching_behavior'].strip())
                    question, options = parse_anthropic_item(raw_item['question'])
                    prompt = f"Question: {question}\nAnswer:"

                    if len(options)>2:
                        # For now skip options with more than two choices
                        # as it makes correct_index meaningless
                        continue

                    items.append(Item(prompt=prompt, options=options,
                        id=item_id))
                    category_features.append(category)
        dataset = Dataset(name=f'anthropic evals - {dataset} - {subdir}', items=items)
        features = [Category(values=category_features)]
        return dataset, features
    else:
        raise NotImplementedError("Given dataset either doesn't exist or is not implemented yet!")
