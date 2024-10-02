import numpy as np
import random

from ...utils import SAMPLER
import logging

log = logging.getLogger(__name__)


class SemSegRandomSampler(object):
    """Random sampler for semantic segmentation datasets."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        self.split = self.dataset.split

    def __len__(self):
        return self.length

    def initialize_with_dataloader(self, dataloader):
        self.length = len(dataloader)

    def get_cloud_sampler(self):

        def gen():
            ids = np.random.permutation(self.length)
            for i in ids:
                yield i

        return gen()

    @staticmethod
    def get_point_sampler():

        def _random_centered_gen(**kwargs):
            pc = kwargs.get("pc", None)
            num_points = kwargs.get("num_points", None)
            search_tree = kwargs.get("search_tree", None)
            if pc is None or num_points is None or search_tree is None:
                raise KeyError(
                    "Please provide pc, num_points, and search_tree \
                    for point_sampler in SemSegRandomSampler"
                )

            center_idx = np.random.choice(len(pc), 1)
            center_point = pc[center_idx, :].reshape(1, -1)

            if pc.shape[0] < num_points:
                diff = num_points - pc.shape[0]
                idxs = np.array(range(pc.shape[0]))
                idxs = list(idxs) + list(random.choices(idxs, k=diff))
                idxs = np.asarray(idxs)
            else:
                idxs = search_tree.query(center_point, k=num_points)[1][0]
            random.shuffle(idxs)
            # pc = pc[idxs]
            return idxs, center_point

        return _random_centered_gen


SAMPLER._register_module(SemSegRandomSampler)


class SemSegRandomClassSampler(object):
    """Random sampler for semantic segmentation datasets."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        self.split = self.dataset.split

    def __len__(self):
        return self.length

    def initialize_with_dataloader(self, dataloader):
        self.length = len(dataloader)

    def get_cloud_sampler(self):

        def gen():
            ids = np.random.permutation(self.length)
            for i in ids:
                yield i

        return gen()

    @staticmethod
    def get_point_sampler():

        def _random_centered_gen(**kwargs):
            pc = kwargs.get("pc", None)
            label = kwargs.get("label", None)
            num_points = kwargs.get("num_points", None)
            search_tree = kwargs.get("search_tree", None)
            sampler = kwargs.get("sampler", None)
            if (
                pc is None
                or num_points is None
                or search_tree is None
                or sampler is None
            ):
                raise KeyError(
                    "Please provide pc, num_points, search_tree and sampler \
                    for point_sampler in SemSegRandomClassSampler"
                )

            # unique_labels = np.unique(label)
            class_list = list(sampler["classes"])
            while True:
                random_class = np.random.choice(class_list, 1)
                log.info(f"Sampled class: {random_class}")
                idx_class = np.where(label == random_class)[0]
                if len(idx_class) == 0:
                    log.info(f"No points for class {random_class}")
                    class_list.remove(random_class)
                    if len(class_list) == 0:
                        center_idx = np.random.choice(len(pc), 1)
                        break
                    else:
                        continue
                center_idx = np.random.choice(idx_class, 1)
                break

            # center_idx = np.random.choice(len(pc), 1)
            center_point = pc[center_idx, :].reshape(1, -1)

            if pc.shape[0] < num_points:
                diff = num_points - pc.shape[0]
                idxs = np.array(range(pc.shape[0]))
                idxs = list(idxs) + list(random.choices(idxs, k=diff))
                idxs = np.asarray(idxs)
            else:
                idxs = search_tree.query(center_point, k=num_points)[1][0]
            random.shuffle(idxs)

            return idxs, center_point

        return _random_centered_gen


SAMPLER._register_module(SemSegRandomClassSampler)
