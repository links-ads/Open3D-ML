import logging
from typing import List
import numpy as np

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import DATASET, Config, get_module
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class SemSupDataset(BaseDataset):

    def __init__(
        self,
        name="SemSupDataset",
        datasets=[],
        class_weights=[],
        num_classes: int = None,
        steps_per_epoch_train: int = 100,
        steps_per_epoch_valid: int = 10,
        test_result_folder: str = "./test",
        use_cache: bool = False,
        cache_dir: str = "./Open3D-ML/logs/cache",
        sub_sample: List[int] = None,
        **kwargs,
    ):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        self.name = name
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        ext_attributes = {
            "steps_per_epoch_train": steps_per_epoch_train,
            "steps_per_epoch_valid": steps_per_epoch_valid,
            "test_result_folder": test_result_folder,
            "cache_dir": cache_dir,
            "class_weights": class_weights,
            "num_classes": num_classes,
            "use_cache": use_cache,
        }
        self.datasets = []

        self.num_classes = num_classes

        # get the datasets FIXME only works with 2 datasets
        source_cfg, target_cfg = datasets
        self.class_map = source_cfg.get("class_map", None)
        source_cls = get_module("dataset", source_cfg["name"])
        s = {**source_cfg, **ext_attributes}
        self.source_dataset = source_cls(**s)

        target_cls = get_module("dataset", target_cfg["name"])
        t = {**target_cfg, **ext_attributes}
        self.target_dataset = target_cls(**t)
        self.cfg = Config(ext_attributes)

        self.train_files = [dataset.train_files for dataset in self.datasets]
        self.val_files = [dataset.val_files for dataset in self.datasets]
        self.test_files = [dataset.test_files for dataset in self.datasets]

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: "Unclassified",
            1: "Ground",
            2: "Vegetation",
            3: "Building",
            4: "Wall",
            5: "Bridge",
            6: "Parking",
            7: "Rail",
            8: "Traffic Road",
            9: "Street Furniture",
            10: "Car",
            11: "Footpath",
            12: "Bike",
            13: "Water",
            14: "Road Marking",
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return SemSupDatasetSplit(self, split=split)

    def get_split_list(self, split, flatten=True):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ["test", "testing"]:
            files = self.test_files
        elif split in ["train", "training"]:
            files = self.train_files
        elif split in ["val", "validation"]:
            files = self.val_files
        elif split in ["all"]:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        if flatten:
            return [f for fl in files for f in fl]
        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        ds_index = attr["ds_index"]
        return self.datasets[ds_index].is_tested(attr)

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        ds_index = attr["ds_index"]
        return self.datasets[ds_index].save_test_result(results, attr)


class SemSupDatasetSplit(ABC):

    def __init__(self, dataset, split="training"):
        self.cfg = dataset.cfg
        self.split = split
        self.dataset = dataset
        self.source_split = self.dataset.source_dataset.get_split(split)
        self.target_split = self.dataset.target_dataset.get_split(split)
        self.class_maps = self.dataset.class_map

        if split in ["test"]:
            sampler_cls = get_module("sampler", "SemSegSpatiallyRegularSampler")
        else:
            sampler_cfg = self.cfg.get("sampler", {"name": "SemSegRandomSampler"})
            sampler_cls = get_module("sampler", sampler_cfg["name"])
        self.sampler = sampler_cls(self)

        log.info(
            "Found {} pointclouds for {} of source".format(
                len(self.source_split), split
            )
        )
        log.info(
            "Found {} pointclouds for {} of target".format(
                len(self.target_split), split
            )
        )

    def __len__(self):
        return len(self.source_split) + len(self.target_split)

    def get_data(self, idx):
        source_data = self.source_split.get_data(idx % len(self.source_split))
        target_data = self.target_split.get_data(idx % len(self.target_split))
        if self.class_maps is not None:
            source_data["label"] = np.vectorize(lambda v: self.class_maps.get(v, v))(
                source_data["label"]
            )
            # FIXME: now target data is always set to 0, eventually set an option in the config
            target_data["label"] = np.vectorize(lambda v: 0)(target_data["label"])

        return {"source": source_data, "target": target_data}

    def get_attr(self, idx):
        source_attr = self.source_split.get_attr(idx % len(self.source_split))
        target_attr = self.target_split.get_attr(idx % len(self.target_split))
        return {"source": source_attr, "target": target_attr}

    def map_index(self, idx, split="train"):
        ds_index = 0
        count = 0

        for dataset_file_list in self.dataset.get_split_list(split, flatten=False):
            dataset_size = len(dataset_file_list)
            if idx < count + dataset_size:
                file_index = idx - count
                return (ds_index, file_index)
            count += dataset_size
            ds_index += 1
        raise ValueError("Wrong external index")


DATASET._register_module(SemSupDataset)
