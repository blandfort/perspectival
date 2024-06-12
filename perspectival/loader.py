import re
import json
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Optional, Tuple, List

from .features import Category, GroundTruth, NamedFeature
from .dataset import Dataset, Item
from .interfaces import Feature



def load_hellaswag(
        split: str = 'train',
        split_type: Optional[str]=None,
    ) -> Tuple[Dataset, List[Feature]]:
    """Load the HellaSwag dataset.

    Also see https://huggingface.co/datasets/Rowan/hellaswag"""
    dataset_name = 'hellaswag'

    def preprocess(text):
        """Preprocess text of a HellaSwag item.

        Taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/78a215e06a3f8bdf2c4be010025ebec34b80ff80/lm_eval/tasks/hellaswag/utils.py"""
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    # Load the dataset
    raw_dataset = load_dataset(dataset_name, split=split)

    items = []
    ground_truth = []

    for ix, element in enumerate(raw_dataset):
        if split_type is not None:
            if element['split_type']!=split_type:
                continue

        ctx = element["ctx_a"] + " " + element["ctx_b"].capitalize()
        prompt = preprocess(element["activity_label"] + ": " + ctx)
        options = [preprocess(ending) for ending in element['endings']]

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


def parse_anthropic_item(raw_item: dict):
    text = raw_item['question']

    # Extract the question if present, otherwise use a default message
    question_match = re.search(r"(Question: )?(.*?)(?=\nChoices:)", text, re.DOTALL)
    question = question_match.group(2) #if question_match else "No question provided."

    # Extract the choices and remove the leading identifiers like (A), (B), etc.
    choices = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\n|\Z)", text)
    choice_list = [(choice[0], choice[1]) for choice in choices]
    choice_letters = [choice[0] for choice in choice_list]
    assert choice_letters==sorted(choice_letters)

    # Get the choices matching and not matching behavior
    matching_behavior = raw_item['answer_matching_behavior'].strip(" ()")
    assert matching_behavior in choice_letters
    choice_matching_behavior = [ix for ix, choice in enumerate(choice_list) if choice[0]==matching_behavior][0]
    not_matching_behavior = raw_item['answer_not_matching_behavior'].strip(" ()")
    assert not_matching_behavior in choice_letters
    choice_not_matching_behavior = [ix for ix, choice in enumerate(choice_list) if choice[0]==not_matching_behavior][0]

    return question, [choice[1] for choice in choice_list], choice_matching_behavior, choice_not_matching_behavior

def load_anthropic_eval_data(repo_path: Path, dataset: str,
        subdir: Optional[str]=None,
    ) -> Tuple[Dataset, List[Feature]]:
    """Load the advanced AI risk dataset from Anthropic

    If repo_path is not an existing directory, this function will attempt
    to download the repository to that path.

    For dataset details, refer to https://github.com/anthropics/evals.
    """
    if not os.path.isdir(repo_path):
        print(f"Downloading repo to {repo_path}...")
        os.system(f"git clone https://github.com/anthropics/evals {repo_path}")
        print("Finished downloading.")

    if dataset=='advanced-ai-risk':
        assert subdir in ['human_generated_evals', 'lm_generated_evals', 'prompts_for_few_shot_generation']

        items = []
        category_features = []
        matching_behavior = []
        not_matching_behavior = []

        dir_path = repo_path / Path(dataset) / Path(subdir)
        for path in os.listdir(dir_path):

            if path.endswith('.jsonl'):
                category = path.rsplit('.', 1)[0]
                raw_items = [json.loads(line) for line in open(dir_path / Path(path), 'r')]

                for ix, raw_item in enumerate(raw_items):
                    item_id = f"{dataset}/{subdir}/{category}/{ix}"

                    question, options, mb, nmb = parse_anthropic_item(raw_item)
                    # NOTE: Our format is slightly different from the one Anthropic
                    # mentions in the repo
                    # (See https://github.com/anthropics/evals/tree/main/advanced-ai-risk)
                    prompt = f"Human: {question}\nAssistant:"

                    items.append(Item(prompt=prompt, options=options,
                        id=item_id))
                    category_features.append(category)
                    matching_behavior.append(mb)
                    not_matching_behavior.append(nmb)
        dataset = Dataset(name=f'anthropic evals - {dataset} - {subdir}', items=items)
        features = [
            Category(values=category_features),
            NamedFeature(values=matching_behavior, name='Choice_matching_behavior'),
            NamedFeature(values=not_matching_behavior, name='Choice_not_matching_behavior'),
        ]
        return dataset, features
    else:
        raise NotImplementedError("Given dataset either doesn't exist or is not implemented yet!")
