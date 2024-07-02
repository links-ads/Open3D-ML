import glob
from pathlib import Path
import os
import logging

import numpy as np
import laspy as lp

from .customdataset import Custom3D
from ..utils import DATASET, get_module


log = logging.getLogger(__name__)


class TurinDataset3DSplit:
    def __init__(
        self,
        dataset,
        split="training",
        scale=np.array([1.0, 1.0, 1.0]),
        offset=np.array([0.0, 0.0, 0.0]),
    ):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset
        self.scale = scale
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
        points = points * self.scale - self.offset
        feat = np.concatenate(
            [
                np.expand_dims(data.red, axis=1),
                np.expand_dims(data.green, axis=1),
                np.expand_dims(data.blue, axis=1),
            ],
            axis=1,
        )
        feat = (feat >> 8).astype(np.float32)
        if self.split != "test":
            labels = np.array(data.classification, dtype=np.int32)
        else:
            labels = np.zeros((points.shape[0]), dtype=np.int32)

        data = {"point": points, "feat": feat, "label": labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace("." + self.dataset.file_format, "")

        attr = {"name": name, "path": str(pc_path), "split": self.split}

        return attr


class TurinDataset3D(Custom3D):
    def __init__(
        self,
        dataset_path,
        name="TurinDataset3D",
        cache_dir="./logs/cache",
        use_cache=False,
        num_points=65536,
        ignored_label_inds=[],
        test_result_folder="./test",
        file_format="las",
        scale=np.array([1.0, 1.0, 1.0]),
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
            **kwargs
        )
        self.file_format = file_format

        self.scale = np.array(scale)
        self.offset = np.array(offset)

        self.train_files = [
            f for f in glob.glob(self.train_dir + "/*." + self.file_format)
        ]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*." + self.file_format)]
        self.test_files = [
            f for f in glob.glob(self.test_dir + "/*." + self.file_format)
        ]

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return TurinDataset3DSplit(
            self, split=split, scale=self.scale, offset=self.offset
        )

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        # TODO update LAS information with the predicted labels
        cfg = self.cfg
        name = attr["name"]
        path = cfg.test_result_folder
        os.makedirs(path, exist_ok=True)

        pred = results["predict_labels"]
        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        las = lp.read(attr["path"])
        assert len(las.x) == len(
            results["predict_labels"]
        ), "Prediction and points are not of the same size"

        las.classification = pred

        store_path = os.path.join(path, name + "." + self.file_format)
        las.write(store_path)


DATASET._register_module(TurinDataset3D)
