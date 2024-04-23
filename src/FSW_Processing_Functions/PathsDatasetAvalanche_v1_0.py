import torch
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import (
    AvalancheDataset, make_avalanche_dataset, make_detection_dataset, make_classification_dataset, 
    make_tensor_classification_dataset, PathsDataset
)
from typing import (
    Callable,
    Generic,
    List,
    Tuple,
    Sequence,
    Optional,
    TypeVar,
    Union,
    Any,
    Dict,
    Mapping,
    Protocol,
)
from PIL import Image
from collections import defaultdict
from functools import partial
from PIL import Image
from pathlib import Path



"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




import torch
import torch.nn as nn
import torch.nn as F
import torch.utils.data as data
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch import rand, randint, Tensor

from pathlib import Path


import torch.utils.data as data

from PIL import Image
import os
import os.path
import dill

from torch import Tensor
from torchvision.transforms.functional import crop

#from avalanche.checkpointing import constructor_based_serialization

#from .transform_groups import XTransform, YTransform
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    Callable,
    Sequence,
    Protocol,
)

from avalanche.benchmarks.utils.transforms import (
    MultiParamCompose,
    TupleTransform,
    MultiParamTransform,
)
"""


# Info: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols
class XComposedTransformDef(Protocol):
    def __call__(self, *input_values: Any) -> Any:
        pass


class XTransformDef(Protocol):
    def __call__(self, input_value: Any) -> Any:
        pass


class YTransformDef(Protocol):
    def __call__(self, input_value: Any) -> Any:
        pass


XTransform = Optional[Union[XTransformDef, XComposedTransformDef]]
YTransform = Optional[YTransformDef]
TransformGroupDef = Union[None, XTransform, Tuple[XTransform, YTransform]]


def identity(x):
    """
    this is used together with partial to replace a lambda function
    that causes pickle to fail
    """
    return x


class TransformGroups:
    """Transformation groups for Avalanche datasets.

    TransformGroups supports preprocessing and augmentation pipelines for
    Avalanche datasets. Transfomations are separated into groups (e.g. `train`
    transforms and `test` transforms), that can be easily switched using the
    `with_transform` method.
    """

    def __init__(
        self,
        transform_groups: Mapping[
            str,
            Union[None, Callable, Sequence[Union[Callable, XTransform, YTransform]]],
        ],
        current_group="train",
    ):
        """Constructor.

        :param transform_groups: A dictionary with group names (string) as keys
            and transformations (pytorch transformations) as values.
        :param current_group: the currently active group.
        """
        self.transform_groups: Dict[
            str, Union[TupleTransform, MultiParamTransform, None]
        ] = dict()
        for group, transform in transform_groups.items():
            norm_transform = _normalize_transform(transform)
            self.transform_groups[group] = norm_transform

        self.current_group = current_group

        if "train" in self.transform_groups:
            if "eval" not in self.transform_groups:
                self.transform_groups["eval"] = self.transform_groups["train"]

        if "train" not in self.transform_groups:
            self.transform_groups["train"] = None

        if "eval" not in self.transform_groups:
            self.transform_groups["eval"] = None

    def __getitem__(self, item):
        return self.transform_groups[item]

    def __setitem__(self, key, value):
        self.transform_groups[key] = _normalize_transform(value)

    def __call__(self, *args, group_name=None):
        """Apply current transformation group to element."""
        element: List[Any] = list(*args)

        if group_name is None:
            curr_t = self.transform_groups[self.current_group]
        else:
            curr_t = self.transform_groups[group_name]
        if curr_t is None:  # empty group
            return element
        elif not isinstance(curr_t, MultiParamTransform):  #
            element[0] = curr_t(element[0])
        else:
            element = curr_t(*element)
        return element

    def __add__(self, other: "TransformGroups"):
        tgroups = {**self.transform_groups}
        for gname, gtrans in other.transform_groups.items():
            if gname not in tgroups:
                tgroups[gname] = gtrans
            elif gtrans is not None:
                composed_transforms = []
                self_group = tgroups[gname]
                other_group = gtrans

                to_expand_group: Union[TupleTransform, MultiParamTransform, None]
                for to_expand_group in [self_group, other_group]:
                    if to_expand_group is None:
                        pass
                    else:
                        assert callable(to_expand_group)
                        composed_transforms.append(to_expand_group)

                tgroups[gname] = MultiParamCompose(composed_transforms)
        return TransformGroups(tgroups, self.current_group)

    def __eq__(self, other: object):
        if not isinstance(other, TransformGroups):
            return NotImplemented
        return (
            self.transform_groups == other.transform_groups
            and self.current_group == other.current_group
        )

    def with_transform(self, group_name):
        assert group_name in self.transform_groups
        self.current_group = group_name

    def __str__(self):
        res = ""
        for k, v in self.transform_groups.items():
            if len(res) > 0:
                res += "\n"
            res += f"- {k}: {v}"
        res = f"current_group: '{self.current_group}'\n" + res
        return res

    def __copy__(self):
        # copy of TransformGroups should copy the dictionary
        # to avoid side effects
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.transform_groups = self.transform_groups.copy()
        return result


class DefaultTransformGroups(TransformGroups):
    """A transformation groups that is equal for all groups."""

    def __init__(self, transform):
        super().__init__({})
        transform = _normalize_transform(transform)
        self.transform_groups = defaultdict(partial(identity, transform))

    def with_transform(self, group_name):
        self.current_group = group_name


class EmptyTransformGroups(DefaultTransformGroups):
    def __init__(self):
        super().__init__({})
        self.transform_groups = defaultdict(partial(identity, None))

    def __call__(self, elem, group_name=None):
        """Apply current transformation group to element."""
        if self.transform_groups[group_name] is None:
            return elem
        else:
            return super().__call__(elem, group_name=group_name)


def _normalize_transform(transforms):
    """Normalize transform to MultiParamTransform."""
    if transforms is None:
        return None
    if not isinstance(transforms, MultiParamTransform):
        if isinstance(transforms, Sequence):
            return TupleTransform(transforms)
        else:
            return TupleTransform([transforms])
    return transforms


__all__ = [
    "XComposedTransformDef",
    "XTransformDef",
    "YTransformDef",
    "XTransform",
    "YTransform",
    "TransformGroupDef",
    "TransformGroups",
    "DefaultTransformGroups",
    "EmptyTransformGroups",
]

def default_image_loader(path):
    """
    Sets the default image loader for the Pytorch Dataset.

    :param path: relative or absolute path of the file to load.

    :returns: Returns the image as a RGB PIL image.
    """
    return Image.open(path).convert("RGB")

T = TypeVar("T", covariant=True)
TTargetsType = TypeVar("TTargetsType")
PathALikeT = Union[Path, str]
CoordsT = Union[int, float]
CropBoxT = Tuple[CoordsT, CoordsT, CoordsT, CoordsT]
FilesDefT = Union[
    Tuple[PathALikeT, TTargetsType], Tuple[PathALikeT, TTargetsType, Sequence[int]]
]

class PathsDataset(Dataset[Tuple[T, TTargetsType]], Generic[T, TTargetsType]):
    """
    This class extends the basic Pytorch Dataset class to handle list of paths
    as the main data source.
    """

    def __init__(
        self,
        root: Optional[PathALikeT],
        files: Sequence[FilesDefT[TTargetsType]],
        transform: XTransform = None,
        target_transform: YTransform = None,
        loader: Callable[[str], T] = default_image_loader,
    ):
        """
        Creates a File Dataset from a list of files and labels.

        :param root: root path where the data to load are stored. May be None.
        :param files: list of tuples. Each tuple must contain two elements: the
            full path to the pattern and its class label. Optionally, the tuple
            may contain a third element describing the bounding box to use for
            cropping (top, left, height, width).
        :param transform: eventual transformation to add to the input data (x)
        :param target_transform: eventual transformation to add to the targets
            (y)
        :param loader: loader function to use (for the real data) given path.
        """

        self.root: Optional[Path] = Path(root) if root is not None else None
        self.imgs = files
        self.targets = [img_data[1] for img_data in self.imgs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Returns next element in the dataset given the current index.

        :param index: index of the data to get.
        :return: loaded item.
        """

        img_description = self.imgs[index]
        impath = img_description[0]
        target = img_description[1]
        bbox = None
        if len(img_description) > 2:
            bbox = img_description[2]

        if self.root is not None:
            impath = self.root / impath
        img = self.loader(impath)

        # If a bounding box is provided, crop the image before passing it to
        # any user-defined transformation.
        if bbox is not None:
            if isinstance(bbox, Tensor):
                bbox = bbox.tolist()
            # crop accepts PIL images, too
            img = crop(img, *bbox)  # type: ignore

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns the total number of elements in the dataset.

        :return: Total number of dataset items.
        """

        return len(self.imgs)

def create_paths_datasets(list_dict, train_transform, test_transform, root=None):
    """
    Creates PathsDataset instances from list_dict, applying the given transformations.
    Handles whether there's a validation split by the length of each list in list_dict.

    Parameters:
    list_dict (dict): Dictionary containing lists of data tuples for different splits.
    train_transform (callable): Transformation to apply to training data.
    test_transform (callable): Transformation to apply to testing (and optionally validation) data.
    root (str, optional): Root directory for the datasets.

    Returns:
    dict: A dictionary with the same structure as list_dict, where each list of data tuples is
          replaced with a corresponding PathsDataset instance.

    # Define your transformations
    train_transform = None  # Replace None with your actual training transformation
    test_transform = None  # Replace None with your actual test/validation transformation
    
    # Create the paths_datasets_dict
    paths_datasets_dict = create_paths_datasets(list_dict, train_transform, test_transform)
    """
    paths_datasets_dict = {}

    for group, data_lists in list_dict.items():
        datasets = []
        for i, data_list in enumerate(data_lists):
            # Determine the appropriate transformation to apply
            transform = train_transform if i == 0 else test_transform
            # Create the PathsDataset instance with the appropriate parameters
            dataset = PathsDataset(root=root, files=data_list, transform=transform)
            datasets.append(dataset)
        paths_datasets_dict[group] = tuple(datasets)

    return paths_datasets_dict

def create_classification_datasets(paths_datasets_dict):
    """
    Applies make_classification_dataset to each dataset in paths_datasets_dict,
    using the group number as task_labels. Handles both scenarios of having training,
    validation, and testing splits or just training and testing splits.

    Parameters:
    paths_datasets_dict (dict): A dictionary similar to paths_datasets_dict containing datasets.

    Returns:
    dict: A dictionary with the same structure as paths_datasets_dict, where each dataset is
          replaced with the output of make_classification_dataset.

    # Example usage:
    # Assuming paths_datasets_dict is your dictionary containing PathsDataset instances for each group
    classification_datasets_dict = create_classification_datasets(paths_datasets_dict)
    """
    classification_datasets_dict = {}

    for group_number, datasets_tuple in paths_datasets_dict.items():
        # Apply make_classification_dataset to each dataset in the tuple
        classification_datasets = []
        for dataset in datasets_tuple:
            # Using the group number as the task_labels
            classification_dataset = make_classification_dataset(dataset, task_labels=group_number)
            classification_datasets.append(classification_dataset)
        classification_datasets_dict[group_number] = tuple(classification_datasets)

    return classification_datasets_dict

def aggregate_datasets_by_phase(classification_datasets_dict):
    """
    Aggregates datasets from classification_datasets_dict into a new dictionary
    with keys for each phase ('train', 'validation' (optional), 'test'), where each key
    contains a list of all groups' datasets for that phase.

    Parameters:
    classification_datasets_dict (dict): Dictionary with groups of datasets, where each group
                                         can have 2 or 3 datasets (train, validation (optional), test).

    Returns:
    dict: Dictionary with keys 'train', 'validation' (optional), 'test'. Each key holds a list of
          datasets from all groups for that phase.

    # Example usage:
    # Assuming classification_datasets_dict is structured with your grouped datasets
    aggregated_datasets = aggregate_datasets_by_phase(classification_datasets_dict)
    
    # Now, aggregated_datasets will have keys 'train', 'validation' (if present), and 'test',
    # with each containing a list of the corresponding datasets from all groups.
    """
    # Initialize the aggregation dictionary with lists
    aggregated_dict = {'train': [], 'test': []}
    # Temporarily assume validation might not exist
    validation_present = False
    
    for group_number, datasets_tuple in classification_datasets_dict.items():
        # Always add train and test datasets
        aggregated_dict['train'].append(datasets_tuple[0])
        aggregated_dict['test'].append(datasets_tuple[-1])  # Last position for compatibility with both scenarios
        
        # Check if validation is present (3 datasets)
        if len(datasets_tuple) == 3:
            validation_present = True
            if 'validation' not in aggregated_dict:
                aggregated_dict['validation'] = []
            aggregated_dict['validation'].append(datasets_tuple[1])
    
    # If validation was not present in any of the groups, remove the key
    if not validation_present:
        aggregated_dict.pop('validation', None)
    
    return aggregated_dict