import glob
from pathlib import Path
import os
import os
import logging

import numpy as np
import laspy as lp

from .customdataset import Custom3D
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import DATASET, get_module


log = logging.getLogger(__name__)


class FRACTALDatasetSplit:
    def __init__(
        self,
        dataset,
        split="training",
        offset=np.array([0.0, 0.0, 0.0]),
    ):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset
        self.offset = offset

        if split in ["test"]:
            sampler_cls = get_module("sampler", "SemSegSpatiallyRegularSampler")
        else:
            sampler_cfg = self.cfg.get("sampler", {"name": "SemSegRandomSampler"})
            sampler_cls = get_module("sampler", sampler_cfg["name"])
        self.sampler = sampler_cls(self)

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        data = lp.read(pc_path)
        points = np.concatenate(
            [
                np.expand_dims(data.x, axis=1),
                np.expand_dims(data.y, axis=1),
                np.expand_dims(data.z, axis=1),
            ],
            axis=1,
        )
        points = points - self.offset
        feat = np.concatenate(
            [
                np.expand_dims(data.red, axis=1),
                np.expand_dims(data.green, axis=1),
                np.expand_dims(data.blue, axis=1),
            ],
            axis=1,
        )
        feat = (feat >> 8).astype(np.float32)
        try:
            labels = data.classification.astype(np.int32)
        except:
            labels = np.zeros((points.shape[0]), dtype=np.int32)

        data = {"point": points, "feat": feat, "label": labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.stem

        attr = {"idx":idx, "name": name, "path": str(pc_path), "split": self.split}

        return attr


class FRACTALDataset(BaseDataset):
    def __init__(
        self,
        dataset_path,
        name="FRACTALDataset",
        cache_dir="./logs/cache",
        use_cache=False,
        num_points=65536,
        ignored_label_inds=[],
        test_result_folder="./test",
        train_dir="train",
        val_dir="val",
        test_dir="test",
        offset=np.array([0.0, 0.0, 0.0]),
        **kwargs
    ):

        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            num_points=num_points,
            ignored_label_inds=ignored_label_inds,
            test_result_folder=test_result_folder,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            **kwargs
        )

        self.offset = np.array(offset)

        base = Path(dataset_path)
        self.train_files = list((base / train_dir).glob("*.laz"))
        self.val_files = list((base / val_dir).glob("*.laz"))
        self.test_files = list((base / test_dir).glob("*.laz"))


    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            1 : "Unclassified",
            2 : "Ground",
            3 : "Low Vegetation",
            4 : "Medium Vegetation",
            5 : "High Vegetation",
            6 : "Building",
            9 : "Water",
            17 : "Bridge",
            64 : "Permament Structure",
            65 : "Artefact",
            66 : "Synthetic",
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
        return FRACTALDatasetSplit(self, split=split, offset=self.offset)

    def get_split_list(self, split):
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

        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr["name"]
        path = cfg.test_result_folder
        store_path = join(path, name + ".laz")
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr["name"]
        path = cfg.test_result_folder
        os.makedirs(path, exist_ok=True)

        pred = results["predict_labels"]
        if cfg.ignored_label_inds is not None and len(cfg.ignored_label_inds) > 0:
            for ign in cfg.ignored_label_inds:
                pred[pred >= ign] += 1

        las = lp.read(attr["path"])
        assert len(las.x) == len(
            results["predict_labels"]
        ), "Prediction and points are not of the same size"

        las.classification = pred

        store_path = os.path.join(path, name + ".laz")
        las.write(store_path)


DATASET._register_module(FRACTALDataset)
