import numpy as np
from numpy.lib.recfunctions import merge_arrays
from pathlib import Path
from os.path import join, exists
import logging
import open3d as o3d
from plyfile import PlyData, PlyElement

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET
import os

log = logging.getLogger(__name__)


class STPLS3D(BaseDataset):
    """STPLS3D dataset, used in visualizer, training, or test."""

    def __init__(
        self,
        dataset_path,
        name="STPLS3D",
        cache_dir="./logs/cache",
        use_cache=False,
        num_points=65536,
        class_weights=None,
        ignored_label_inds=[],
        train_files=[],
        val_files=[],
        test_files=[],
        test_result_folder="./test",
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
        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            class_weights=class_weights,
            num_points=num_points,
            ignored_label_inds=ignored_label_inds,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            test_result_folder=test_result_folder,
            **kwargs,
        )

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()

        self.dataset_path = cfg.dataset_path
        self.num_classes = kwargs.get("num_classes", len(self.label_to_names))
        if class_weights is None:
            self.class_weights = np.ones(self.num_classes, dtype=np.float32)
        else:
            self.class_weights = np.array(class_weights, dtype=np.float32)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_files = [join(self.cfg.dataset_path, f) for f in train_files]
        self.val_files = [join(self.cfg.dataset_path, f) for f in val_files]
        self.test_files = [join(self.cfg.dataset_path, f) for f in test_files]

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: "Ground",
            1: "Building",
            2: "LowVegetation",
            3: "MediumVegetation",
            4: "HighVegetation",
            5: "Vehicle",
            6: "Truck",
            7: "Aircraft",
            8: "MilitaryVehicle",
            9: "Bike",
            10: "Motorcycle",
            11: "LightPole",
            12: "StreetSgin",
            13: "Clutter",
            14: "Fence",
            15: "Road",
            17: "Windows",
            18: "Dirt",
            19: "Grass",
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
        return STPLS3DSplit(self, split=split)

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
        store_path = join(path, self.name, name + ".ply")
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr, save_features=False):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr["name"].split(".")[0]
        path = os.path.join(cfg.test_result_folder, cfg.experiment)
        os.makedirs(path, exist_ok=True)

        pred = results["predict_labels"]
        pred = np.array(pred, dtype=[("class_pred", "u1")])
        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        plydata = PlyData.read(attr["path"])
        prop_names = [p.name for p in plydata.elements[0].properties]
        if "class_pred" not in prop_names:
            a = merge_arrays([plydata.elements[0].data, pred], flatten=True)
            v = PlyElement.describe(a, "vertex")
            plydata = PlyData([v], text=True)
        else:
            plydata.elements[0].data["class_pred"] = pred
        plydata.write(join(path, name + ".ply"))

        if save_features:
            feat = results["predict_features"]
            assert len(plydata.elements[0].data) == len(
                feat
            ), "Features and points are not of the same size"
            np.save(os.path.join(path, name + ".npy"), feat)


class STPLS3DSplit(BaseDatasetSplit):

    def __init__(self, dataset, split="training"):
        super().__init__(dataset, split=split)

        log.info("Found {} pointclouds for {}".format(len(self.path_list), split))
        self.offset = np.array([0, 0, 0])

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        data = o3d.t.io.read_point_cloud(pc_path).point

        points = data["positions"].numpy() - self.offset
        points = np.float32(points)
        assert not np.isnan(points).any(), f"Nan points in {pc_path}"
        feat = data["colors"].numpy().astype(np.float32)

        try:
            labels = data["class"].numpy().astype(np.int32).reshape((-1,))
        except:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {"point": points, "feat": feat, "label": labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace(".txt", "")

        pc_path = str(pc_path)
        split = self.split
        attr = {"idx": idx, "name": name, "path": pc_path, "split": split}

        return attr


DATASET._register_module(STPLS3D)
