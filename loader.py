import re
import json
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Optional

from features import Category
from dataset import Dataset, Item
from experiment import Experiment


def load_rotten_tomatoes(shuffle: bool = True, split: str = 'train') -> Dataset:
    dataset_name = "rotten_tomatoes"

    # Load the dataset
    raw_dataset = load_dataset(dataset_name, split=split)

    items = []

    # Define options for sentiment based on the labels used in the dataset
    options = ['negative', 'positive']  # Define options for sentiment

    for ix, element in enumerate(raw_dataset):
        prompt = f"Question: What is the sentiment of the text below?\n\nText: '{element['text']}'\n\nAnswer: The sentiment is"
        correct_index = element['label']  # Label directly corresponds to the index in options
        item_id = f"{split}_{ix}"
        items.append(Item(id=item_id, prompt=prompt, options=options, correct_index=correct_index))

    if shuffle:
        np.random.shuffle(items)  # Shuffle the list of items to randomize the order

    # Create and return a Dataset instance
    dataset = Dataset(name=dataset_name, items=items)
    return dataset


def parse_anthropic_item(text, answer_not_matching_behavior):
    # Extract the question if present, otherwise use a default message
    question_match = re.search(r"(Question: )?(.*?)(?=\nChoices:)", text, re.DOTALL)
    question = question_match.group(2) #if question_match else "No question provided."

    # Extract the choices and remove the leading identifiers like (A), (B), etc.
    choices = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\n|\Z)", text)
    choice_list = [(choice[0], choice[1]) for choice in choices]
    choice_letters = [choice[0] for choice in choice_list]
    assert choice_letters==sorted(choice_letters)

    if answer_not_matching_behavior.strip()=='(A)':
        unproblematic_index = 0
    elif answer_not_matching_behavior.strip()=='(B)':
        unproblematic_index = 1
    else:
        #TODO If this happens, need to modify such that
        # we match letters and get the corresponding index
        raise ValueError("Invalid answer index specified!")

    return question, [choice[1] for choice in choice_list], unproblematic_index

def load_anthropic_eval_data(repo_path: Path, dataset: str,
        subdir: Optional[str]=None,
    ) -> dict:
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
                    question, options, correct_index = parse_anthropic_item(raw_item['question'],
                        raw_item['answer_not_matching_behavior'])
                    prompt = f"Question: {question}\nAnswer:"

                    if len(options)>2:
                        # For now skip options with more than two choices
                        # as it makes correct_index meaningless
                        continue

                    items.append(Item(prompt=prompt, options=options,
                        correct_index=correct_index, id=item_id))
                    category_features.append(category)
        dataset = Dataset(name=f'anthropic evals - {dataset} - {subdir}', items=items)
        categories = Category(values=category_features)
        return {'dataset': dataset, 'features': [categories]}
    else:
        raise NotImplementedError("Given dataset either doesn't exist or is not implemented yet!")
