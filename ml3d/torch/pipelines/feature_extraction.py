import logging
from os.path import exists, join
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# pylint: disable-next=unused-import
from open3d.visualization.tensorboard_plugin import summary
from .base_pipeline import BasePipeline
from ..dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher
from ..utils import latest_torch_ckpt
from ..modules.losses import SemSegLoss, filter_valid_label
from ..modules.metrics import SemSegMetric
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets import InferenceDummySplit

log = logging.getLogger(__name__)


class FeatureExtraction(BasePipeline):
    """This class allows you to perform semantic segmentation for both training
    and inference using the Torch. This pipeline has multiple stages: Pre-
    processing, loading dataset, testing, and inference or training.

    **Example:**
        This example loads the Semantic Segmentation and performs a training
        using the SemanticKITTI dataset.

            import torch
            import torch.nn as nn

            from .base_pipeline import BasePipeline
            from torch.utils.tensorboard import SummaryWriter
            from ..dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher

            Mydataset = TorchDataloader(dataset=dataset.get_split('training')),
            MyModel = SemanticSegmentation(self,model,dataset=Mydataset, name='SemanticSegmentation',
            name='MySemanticSegmentation',
            batch_size=4,
            val_batch_size=4,
            test_batch_size=3,
            max_epoch=100,
            learning_rate=1e-2,
            lr_decays=0.95,
            save_ckpt_freq=20,
            adam_lr=1e-2,
            scheduler_gamma=0.95,
            momentum=0.98,
            main_log_dir='./logs/',
            device='gpu',
            split='train')

    **Args:**
            dataset: The 3D ML dataset class. You can use the base dataset, sample datasets , or a custom dataset.
            model: The model to be used for building the pipeline.
            name: The name of the current training.
            batch_size: The batch size to be used for training.
            val_batch_size: The batch size to be used for validation.
            test_batch_size: The batch size to be used for testing.
            max_epoch: The maximum size of the epoch to be used for training.
            leanring_rate: The hyperparameter that controls the weights during training. Also, known as step size.
            lr_decays: The learning rate decay for the training.
            save_ckpt_freq: The frequency in which the checkpoint should be saved.
            adam_lr: The leanring rate to be applied for Adam optimization.
            scheduler_gamma: The decaying factor associated with the scheduler.
            momentum: The momentum that accelerates the training rate schedule.
            main_log_dir: The directory where logs are stored.
            device: The device to be used for training.
            split: The dataset split to be used. In this example, we have used "train".

    **Returns:**
            class: The corresponding class.
    """

    def __init__(
        self,
        model,
        dataset=None,
        name="FeatureExtraction",
        batch_size=4,
        val_batch_size=4,
        test_batch_size=3,
        max_epoch=100,  # maximum epoch during training
        learning_rate=1e-2,  # initial learning rate
        lr_decays=0.95,
        save_ckpt_freq=20,
        adam_lr=1e-2,
        scheduler_gamma=0.95,
        momentum=0.98,
        main_log_dir="./logs/",
        device="cuda",
        split="train",
        **kwargs,
    ):

        super().__init__(
            model=model,
            dataset=dataset,
            name=name,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            max_epoch=max_epoch,
            learning_rate=learning_rate,
            lr_decays=lr_decays,
            save_ckpt_freq=save_ckpt_freq,
            adam_lr=adam_lr,
            scheduler_gamma=scheduler_gamma,
            momentum=momentum,
            main_log_dir=main_log_dir,
            device=device,
            split=split,
            **kwargs,
        )

    def run_inference(self, data):
        """Run inference on given data.

        Args:
            data: A raw data.

        Returns:
            Returns the inference results.
        """
        cfg = self.cfg
        model = self.model
        device = self.device

        model.to(device)
        model.device = device
        model.eval()

        batcher = self.get_batcher(device)
        infer_dataset = InferenceDummySplit(data)
        self.dataset_split = infer_dataset
        infer_sampler = infer_dataset.sampler
        infer_split = TorchDataloader(
            dataset=infer_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=infer_sampler,
            use_cache=False,
        )
        infer_loader = DataLoader(
            infer_split,
            batch_size=cfg.batch_size,
            sampler=get_sampler(infer_sampler),
            collate_fn=batcher.collate_fn,
        )

        model.trans_point_sampler = infer_sampler.get_point_sampler()
        self.curr_cloud_id = -1
        self.test_probs = []
        self.ori_test_probs = []
        self.ori_test_labels = []
        self.ori_test_feats = []

        with torch.no_grad():
            for unused_step, inputs in enumerate(infer_loader):
                results, feats = model(inputs["data"])  # list of features
                self.update_tests(infer_sampler, inputs, results, feats)

        inference_result = {
            "predict_labels": self.ori_test_labels.pop(),
            "predict_scores": self.ori_test_probs.pop(),
            "predict_features": self.ori_test_feats.pop(),
        }

        return inference_result

    def run_test(self):
        """Run the test using the data passed."""
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)
        model.eval()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, "log_test_" + timestamp + ".txt")
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = self.get_batcher(device)

        test_dataset = dataset.get_split("test")
        test_sampler = test_dataset.sampler
        test_split = TorchDataloader(
            dataset=test_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=test_sampler,
            use_cache=dataset.cfg.use_cache,
        )
        test_loader = DataLoader(
            test_split,
            batch_size=cfg.test_batch_size,
            sampler=get_sampler(test_sampler),
            collate_fn=batcher.collate_fn,
        )

        self.dataset_split = test_dataset

        self.load_ckpt(model.cfg.ckpt_path)

        model.trans_point_sampler = test_sampler.get_point_sampler()
        self.curr_cloud_id = -1
        self.test_probs = []
        self.test_feats = []
        self.ori_test_probs = []
        self.ori_test_labels = []
        self.ori_test_feats = []

        log.info("Started testing")

        with torch.no_grad():
            for unused_step, inputs in enumerate(test_loader):
                if hasattr(inputs["data"], "to"):
                    inputs["data"].to(device)
                results = model(inputs["data"])
                class_prob, feats = results[0], results[1]
                self.update_tests(test_sampler, inputs, class_prob, feats)

                if self.complete_infer:
                    inference_result = {
                        "predict_labels": self.ori_test_labels.pop(),
                        "predict_scores": self.ori_test_probs.pop(),
                        "predict_features": self.ori_test_feats.pop(),
                    }
                    attr = self.dataset_split.get_attr(test_sampler.cloud_id)
                    dataset.save_test_result(inference_result, attr, save_features=True)
        try:
            log.info(
                f"Overall Testing Accuracy : {self.metric_test.acc()[-1]}, mIoU : {self.metric_test.iou()[-1]}"
            )
        except:
            log.info(f"Cannot estimate overall accuracy and IoU")

        log.info("Finished testing")

    def update_tests(self, sampler, inputs, results, feats):
        """Update tests using sampler, inputs, and results."""
        split = sampler.split
        end_threshold = 0.5
        if self.curr_cloud_id != sampler.cloud_id:
            self.curr_cloud_id = sampler.cloud_id
            num_points = sampler.possibilities[sampler.cloud_id].shape[0]
            self.pbar = tqdm(
                total=num_points,
                desc="{} {}/{}".format(split, self.curr_cloud_id, len(sampler.dataset)),
            )
            self.pbar_update = 0
            self.test_probs.append(
                np.zeros(
                    shape=[num_points, self.model.cfg.num_classes], dtype=np.float16
                )
            )
            self.test_feats.append(
                np.zeros(
                    shape=[num_points, self.model.cfg.dim_output[0] * 2],
                    dtype=np.float16,
                )
            )
            self.complete_infer = False

        this_possiblility = sampler.possibilities[sampler.cloud_id]
        self.pbar.update(
            this_possiblility[this_possiblility > end_threshold].shape[0]
            - self.pbar_update
        )
        self.pbar_update = this_possiblility[this_possiblility > end_threshold].shape[0]
        self.test_probs[self.curr_cloud_id] = self.model.update_probs(
            inputs,
            results,
            self.test_probs[self.curr_cloud_id],
        )
        self.test_feats[self.curr_cloud_id] = self.model.update_feats(
            inputs, feats, self.test_feats[self.curr_cloud_id]
        )

        if (
            split in ["test"]
            and this_possiblility[this_possiblility > end_threshold].shape[0]
            == this_possiblility.shape[0]
        ):

            proj_inds = self.model.preprocess(
                self.dataset_split.get_data(self.curr_cloud_id), {"split": split}
            ).get("proj_inds", None)
            if proj_inds is None:
                proj_inds = np.arange(self.test_probs[self.curr_cloud_id].shape[0])
            test_labels = np.argmax(self.test_probs[self.curr_cloud_id][proj_inds], 1)
            test_feats = self.test_feats[self.curr_cloud_id][proj_inds]
            # test features are not reduced along last dimension
            self.ori_test_probs.append(self.test_probs[self.curr_cloud_id][proj_inds])
            self.ori_test_labels.append(test_labels)
            self.ori_test_feats.append(test_feats)
            self.complete_infer = True

    def run_train(self):
        raise NotImplementedError

    def get_batcher(self, device, split="training"):
        """Get the batcher to be used based on the device and split."""
        batcher_name = getattr(self.model.cfg, "batcher")

        if batcher_name == "DefaultBatcher":
            batcher = DefaultBatcher()
        elif batcher_name == "ConcatBatcher":
            batcher = ConcatBatcher(device, self.model.cfg.name)
        else:
            batcher = None
        return batcher

    def get_3d_summary(self, results, input_data, epoch, save_gt=True):
        raise NotImplementedError

    def save_logs(self, writer, epoch):
        """Save logs from the training and send results to TensorBoard."""
        train_accs = self.metric_train.acc()
        val_accs = self.metric_val.acc()

        train_ious = self.metric_train.iou()
        val_ious = self.metric_val.iou()

        loss_dict = {
            "Training loss": np.mean(self.losses),
            "Validation loss": np.mean(self.valid_losses),
        }
        acc_dicts = [
            {"Training accuracy": acc, "Validation accuracy": val_acc}
            for acc, val_acc in zip(train_accs, val_accs)
        ]

        iou_dicts = [
            {"Training IoU": iou, "Validation IoU": val_iou}
            for iou, val_iou in zip(train_ious, val_ious)
        ]

        for key, val in loss_dict.items():
            writer.add_scalar(key, val, epoch)
        for key, val in acc_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)
        for key, val in iou_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)

        log.info(
            f"Loss train: {loss_dict['Training loss']:.3f} "
            f" eval: {loss_dict['Validation loss']:.3f}"
        )
        log.info(
            f"Mean acc train: {acc_dicts[-1]['Training accuracy']:.3f} "
            f" eval: {acc_dicts[-1]['Validation accuracy']:.3f}"
        )
        log.info(
            f"Mean IoU train: {iou_dicts[-1]['Training IoU']:.3f} "
            f" eval: {iou_dicts[-1]['Validation IoU']:.3f}"
        )

        for stage in self.summary:
            for key, summary_dict in self.summary[stage].items():
                label_to_names = summary_dict.pop("label_to_names", None)
                writer.add_3d(
                    "/".join((stage, key)),
                    summary_dict,
                    epoch,
                    max_outputs=0,
                    label_to_names=label_to_names,
                )

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        """Load a checkpoint. You must pass the checkpoint and indicate if you
        want to resume.
        """
        train_ckpt_dir = join(self.cfg.logs_dir, "checkpoint")
        make_dir(train_ckpt_dir)

        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info("ckpt_path not given. Restore from the latest ckpt")
            else:
                log.info("Initializing from scratch.")
                return

        if not exists(ckpt_path):
            raise FileNotFoundError(f" ckpt {ckpt_path} not found")

        log.info(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and hasattr(self, "optimizer"):
            log.info(f"Loading checkpoint optimizer_state_dict")
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and hasattr(self, "scheduler"):
            log.info(f"Loading checkpoint scheduler_state_dict")
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    def save_ckpt(self, epoch):
        """Save a checkpoint at the passed epoch."""
        path_ckpt = join(self.cfg.logs_dir, "checkpoint")
        make_dir(path_ckpt)
        torch.save(
            dict(
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
            ),
            join(path_ckpt, f"ckpt_{epoch:05d}.pth"),
        )
        log.info(f"Epoch {epoch:3d}: save ckpt to {path_ckpt:s}")

    def save_config(self, writer):
        """Save experiment configuration with tensorboard summary."""
        if hasattr(self, "cfg_tb"):
            writer.add_text("Description/Open3D-ML", self.cfg_tb["readme"], 0)
            writer.add_text("Description/Command line", self.cfg_tb["cmd_line"], 0)
            writer.add_text(
                "Configuration/Dataset",
                code2md(self.cfg_tb["dataset"], language="json"),
                0,
            )
            writer.add_text(
                "Configuration/Model", code2md(self.cfg_tb["model"], language="json"), 0
            )
            writer.add_text(
                "Configuration/Pipeline",
                code2md(self.cfg_tb["pipeline"], language="json"),
                0,
            )


PIPELINE._register_module(FeatureExtraction, "torch")
