from typing import Optional, List

from dataset import Dataset
from features import Feature


def display_item(dataset: Dataset, item_ix: int,
                 features: Optional[List[str]]=None,
                 predictions: Optional[List[str]]=None,
                 comparisons: Optional[List[str]]=None,
    ):
    item = dataset.items[item_ix]
    print(f"ITEM ({item.id})")
    print(f'"""{item.prompt}"""')
    print(f"Options: {item.options}")
    print(f"Ground truth: {item.options[item.correct_index]}")

    if features is not None:
        print("\nFEATURES")
        for feature_cls in features:
            feature_dict = dataset.get_features(feature_cls)
            for key in feature_dict:
                print(key, feature_dict[key].values[item_ix])

def display_items(dataset: Dataset,
                item_ixs: Optional[List[int]]=None,
                features: Optional[List[str]]=None,
                predictions: Optional[List[str]]=None,
                comparisons: Optional[List[str]]=None,
    ):
    if item_ixs is None:
        item_ixs = range(len(dataset.items))
    for ix in item_ixs:
        display_item(dataset=dataset, item_ix=ix, features=features, predictions=predictions, comparisons=comparisons)
        print('\n\n' + 20*'-' + '\n\n')
