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
from ..dataloaders import (
    get_sampler,
    TorchDataloader,
    DefaultBatcher,
    ConcatBatcher,
    SemSupDataloader,
)
from ..utils import latest_torch_ckpt
from ..modules.losses import SemSegLoss, filter_valid_label, SoftmaxEntropyLoss
from ..modules.metrics import SemSegMetric
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets import InferenceDummySplit
from torch import nn
import torch.nn.functional as F
import open3d.ml as _ml3d

log = logging.getLogger(__name__)


class SemiSupervisedSegmentation(BasePipeline):
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
        name="SemanticSegmentation",
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
        self.student_model = model
        Model = _ml3d.utils.get_module("model", model.cfg.cfg_dict["name"], "torch")
        self.teacher_model = Model(**model.cfg)

    def run_inference(self, data):
        raise NotImplementedError

    def run_test(self):
        raise NotImplementedError

    def update_tests(self, sampler, inputs, results):
        raise NotImplementedError

    def run_train(self):

        torch.manual_seed(
            self.rng.integers(np.iinfo(np.int32).max)
        )  # Random reproducible seed for torch
        student_model = self.student_model
        teacher_model = self.teacher_model
        device = self.device
        student_model.device = device
        teacher_model.device = device
        dataset = self.dataset
        cfg = self.cfg
        student_model.to(device)
        teacher_model.to(device)

        log.info("DEVICE : {}".format(device))
        log.info("Using EMA from epoch {}".format(cfg.ema_start_epoch))
        log.info("Using confidence threshold of {}".format(cfg.confidence_th))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        log_file_path = join(cfg.logs_dir, "log_train_" + timestamp + ".txt")
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))
        # Teacher should produce the pseudolabels from target domain
        # Train student network on mixed data
        # At each batch iteration, we do the following:
        # Inference on the target domain
        # Select only the most confident points from target domain
        # Update the student with the total loss, where only
        # the most confident points from target domain are used in loss
        # the loss is just a Dice

        # Lastly, the teacher is update with moving average
        loss_ce = SemSegLoss(self, student_model, dataset, device)
        # Metrics to be used on mixed data? TODO: check
        self.metric_train = SemSegMetric()

        self.batcher = self.get_batcher(device)

        train_dataset = dataset.get_split("train")
        train_sampler = train_dataset.sampler
        train_split = SemSupDataloader(
            dataset=train_dataset,
            preprocess=student_model.preprocess,
            transform=student_model.transform,
            sampler=train_sampler,
            use_cache=dataset.cfg.use_cache,
            steps_per_epoch=dataset.cfg.get("steps_per_epoch_train", None),
        )

        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            sampler=get_sampler(train_sampler),
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(
                x + np.uint32(torch.utils.data.get_worker_info().seed)
            ),
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        valid_dataset = dataset.get_split("validation")
        valid_sampler = valid_dataset.sampler
        valid_split = SemSupDataloader(
            dataset=valid_dataset,
            preprocess=student_model.preprocess,
            transform=student_model.transform,
            sampler=valid_sampler,
            use_cache=dataset.cfg.use_cache,
            steps_per_epoch=dataset.cfg.get("steps_per_epoch_valid", None),
        )

        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            sampler=get_sampler(valid_sampler),
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(
                x + np.uint32(torch.utils.data.get_worker_info().seed)
            ),
        )

        self.optimizer, self.scheduler = student_model.get_optimizer(cfg)

        is_resume = student_model.cfg.get("is_resume", True)
        self.load_ckpt(
            student_model.cfg.ckpt_path,
            is_resume=is_resume,
        )
        self.load_ckpt(
            teacher_model.cfg.ckpt_path,
            is_resume=is_resume,
        )

        writer = SummaryWriter(self.cfg.logs_dir)
        self.save_config(writer)
        log.info("Writing summary in {}.".format(self.cfg.logs_dir))
        record_summary = cfg.get("summary").get("record_for", [])

        for epoch in range(0, cfg.max_epoch + 1):

            log.info(f"=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===")
            student_model.train()
            teacher_model.train()
            self.metric_train.reset()
            self.losses = []
            student_model.trans_point_sampler = train_sampler.get_point_sampler()
            teacher_model.trans_point_sampler = train_sampler.get_point_sampler()

            for step, inputs in enumerate(tqdm(train_loader, desc="training")):
                # FIXME: move in forward
                if hasattr(inputs["data"]["mixed"], "to"):
                    inputs["data"]["mixed"].to(device)
                if hasattr(inputs["data"]["target"], "to"):
                    inputs["data"]["target"].to(device)

                self.optimizer.zero_grad()
                inputs_mixed = inputs["data"]["mixed"]
                inputs_target = inputs["data"]["target"]
                log.info(
                    f"Num. water points: {torch.sum(inputs_mixed['labels'] == 5).item()}"
                )
                # Get indexes from features
                inputs_mixed["features"], mixed_idx, pointcloud_idx = (
                    inputs_mixed["features"][:, :, :-2],
                    inputs_mixed["features"][:, :, -2],
                    inputs_mixed["features"][:, :, -1],
                )
                pointcloud_idx = pointcloud_idx.bool().cpu()  # TO INT
                mixed_idx = mixed_idx.to(torch.int32).cpu()  # TO INT

                inputs_target["features"], target_idx = (
                    inputs_target["features"][:, :, :-2],
                    inputs_target["features"][:, :, -2],
                )

                target_idx = target_idx.to(torch.int32).cpu()  # TO INT # TO CPU
                # Inference of target pointcloud to get pseudo labels
                if epoch > cfg.get("ema_start_epoch", 10):
                    with torch.no_grad():
                        results_target = teacher_model(inputs_target).cpu()
                        softmax_target = F.softmax(results_target, dim=-1)
                        confidence_pseudolabels_target, pseudolabels_target = (
                            softmax_target.max(dim=-1)
                        )

                        if (
                            student_model.cfg.ignored_label_inds is not None
                            and len(student_model.cfg.ignored_label_inds) > 0
                        ):
                            for ign in student_model.cfg.ignored_label_inds:
                                pseudolabels_target[pseudolabels_target >= ign] += 1
                        pseudolabels_target[
                            confidence_pseudolabels_target < cfg.confidence_th
                        ] = 0
                        log.info(
                            f"Percentage of kept target pseudolabels: {torch.sum(pseudolabels_target!=0).item() / np.prod(pseudolabels_target.shape)}"
                        )

                        # Impute pseudolabels in mixed pointcloud
                        for batch_idx in range(results_target.shape[0]):
                            target_idx_list = target_idx[batch_idx].tolist()
                            pseudolabels_target_list = pseudolabels_target[
                                batch_idx
                            ].tolist()

                            index_to_pseudolabel = dict(
                                zip(
                                    target_idx_list,
                                    pseudolabels_target_list,
                                )
                            )
                            mixed_idx_target = mixed_idx[batch_idx][
                                pointcloud_idx[batch_idx]
                            ].tolist()
                            pseudolabels_mixed = torch.as_tensor(
                                [index_to_pseudolabel[idx] for idx in mixed_idx_target]
                            ).long()

                            inputs_mixed["labels"][batch_idx][
                                pointcloud_idx[batch_idx]
                            ] = pseudolabels_mixed
                    # Get mixed pointcloud inference from student model
                results_mixed = student_model(inputs_mixed)
                # Get loss value
                loss, gt_labels, predict_scores = student_model.get_loss(
                    loss_ce, results_mixed, inputs_mixed, device
                )

                if predict_scores.size()[-1] == 0:
                    continue

                loss.backward()
                if student_model.cfg.get("grad_clip_norm", -1) > 0:
                    torch.nn.utils.clip_grad_value_(
                        student_model.parameters(), student_model.cfg.grad_clip_norm
                    )
                self.optimizer.step()
                # now update the teacher model with EMA
                if epoch > cfg.get("ema_start_epoch", 10):
                    teacher_model.update_ema(student_model)

                self.metric_train.update(predict_scores, gt_labels)

                self.losses.append(loss.cpu().item())
                # Save only for the first pcd in batch
                if "train" in record_summary and step == 0:
                    self.summary["train"] = self.get_3d_summary(
                        results, inputs["data"], epoch
                    )
            self.scheduler.step()

            # --------------------- validation
            student_model.eval()
            self.valid_losses = []
            student_model.trans_point_sampler = valid_sampler.get_point_sampler()
            with torch.no_grad():
                for step, inputs in enumerate(tqdm(valid_loader, desc="validation")):
                    if hasattr(inputs["data"]["mixed"], "to"):
                        inputs["data"]["mixed"].to(device)

                    inputs_mixed = inputs["data"]["mixed"]

                    # Get indexes from features
                    inputs_mixed["features"], mixed_idx, pointcloud_idx = (
                        inputs_mixed["features"][:, :, :-2],
                        inputs_mixed["features"][:, :, -2],
                        inputs_mixed["features"][:, :, -1],
                    )

                    results = student_model(inputs_mixed)
                    # Target values should be ignored
                    loss, gt_labels, predict_scores = student_model.get_loss(
                        loss_ce, results, inputs_mixed, device
                    )

                    if predict_scores.size()[-1] == 0:
                        continue

                    self.valid_losses.append(loss.cpu().item())
                    # Save only for the first batch
                    if "valid" in record_summary and step == 0:
                        self.summary["valid"] = self.get_3d_summary(
                            results, inputs_mixed, epoch
                        )

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0 or epoch == cfg.max_epoch:
                self.save_ckpt(epoch)

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
        """
        Create visualization for network inputs and outputs.

        Args:
            results: Model output (see below).
            input_data: Model input (see below).
            epoch (int): step
            save_gt (bool): Save ground truth (for 'train' or 'valid' stages).

        RandLaNet:
            results (Tensor(B, N, C)): Prediction scores for all classes
            inputs_batch: Batch of pointclouds and labels as a Dict with keys:
                'xyz': First element is Tensor(B,N,3) points
                'labels': (B, N) (optional) labels

        SparseConvUNet:
            results (Tensor(SN, C)): Prediction scores for all classes. SN is
                total points in the batch.
            input_batch (Dict): Batch of pointclouds and labels. Keys should be:
                'point' [Tensor(SN,3), float]: Concatenated points.
                'batch_lengths' [Tensor(B,), int]: Number of points in each
                    point cloud of the batch.
                'label' [Tensor(SN,) (optional)]: Concatenated labels.

        Returns:
            [Dict] visualizations of inputs and outputs suitable to save as an
                Open3D for TensorBoard summary.
        """
        if not hasattr(self, "_first_step"):
            self._first_step = epoch
        label_to_names = self.dataset.get_label_to_names()
        cfg = self.cfg.get("summary")
        max_pts = cfg.get("max_pts")
        if max_pts is None:
            max_pts = np.iinfo(np.int32).max
        use_reference = cfg.get("use_reference", False)
        max_outputs = cfg.get("max_outputs", 1)
        input_pcd = []
        gt_labels = []
        predict_labels = []

        def to_sum_fmt(tensor, add_dims=(0, 0), dtype=torch.int32):
            sten = tensor.cpu().detach().type(dtype)
            new_shape = (1,) * add_dims[0] + sten.shape + (1,) * add_dims[1]
            return sten.reshape(new_shape)

        # Variable size point clouds
        if self.model.cfg["name"] in ("KPFCNN", "KPConv"):
            batch_lengths = input_data.lengths[0].detach().numpy()
            row_splits = np.hstack(((0,), np.cumsum(batch_lengths)))
            max_outputs = min(max_outputs, len(row_splits) - 1)
            for k in range(max_outputs):
                blen_k = row_splits[k + 1] - row_splits[k]
                pcd_step = int(np.ceil(blen_k / min(max_pts, blen_k)))
                res_pcd = results[row_splits[k] : row_splits[k + 1] : pcd_step, :]
                predict_labels.append(to_sum_fmt(torch.argmax(res_pcd, 1), (0, 1)))
                if self._first_step != epoch and use_reference:
                    continue
                pointcloud = input_data.points[0][
                    row_splits[k] : row_splits[k + 1] : pcd_step
                ]
                input_pcd.append(to_sum_fmt(pointcloud[:, :3], (0, 0), torch.float32))
                if torch.any(input_data.labels != 0):
                    gtl = input_data.labels[row_splits[k] : row_splits[k + 1]]
                    gt_labels.append(to_sum_fmt(gtl, (0, 1)))

        elif self.model.cfg["name"] in ("SparseConvUnet", "PointTransformer"):
            if self.model.cfg["name"] == "SparseConvUnet":
                row_splits = np.hstack(((0,), np.cumsum(input_data.batch_lengths)))
            else:
                row_splits = input_data.row_splits
            max_outputs = min(max_outputs, len(row_splits) - 1)
            for k in range(max_outputs):
                blen_k = row_splits[k + 1] - row_splits[k]
                pcd_step = int(np.ceil(blen_k / min(max_pts, blen_k)))
                res_pcd = results[row_splits[k] : row_splits[k + 1] : pcd_step, :]
                predict_labels.append(to_sum_fmt(torch.argmax(res_pcd, 1), (0, 1)))
                if self._first_step != epoch and use_reference:
                    continue
                if self.model.cfg["name"] == "SparseConvUnet":
                    pointcloud = input_data.point[k]
                else:
                    pointcloud = input_data.point[
                        row_splits[k] : row_splits[k + 1] : pcd_step
                    ]
                input_pcd.append(to_sum_fmt(pointcloud[:, :3], (0, 0), torch.float32))
                if getattr(input_data, "label", None) is not None:
                    if self.model.cfg["name"] == "SparseConvUnet":
                        gtl = input_data.label[k]
                    else:
                        gtl = input_data.label[
                            row_splits[k] : row_splits[k + 1] : pcd_step
                        ]
                    gt_labels.append(to_sum_fmt(gtl, (0, 1)))
        # Fixed size point clouds
        elif self.model.cfg["name"] in (
            "RandLANet",
            "RandLANetMixer",
            "PVCNN",
        ):  # Tuple input
            if (
                self.model.cfg["name"] == "RandLANet"
                or self.model.cfg["name"] == "RandLANetMixer"
            ):
                pointcloud = input_data["coords"][0]  # 0 => input to first layer
            elif self.model.cfg["name"] == "PVCNN":
                pointcloud = input_data["point"].transpose(1, 2)
            pcd_step = int(
                np.ceil(pointcloud.shape[1] / min(max_pts, pointcloud.shape[1]))
            )
            predict_labels = to_sum_fmt(
                torch.argmax(results[:max_outputs, ::pcd_step, :], 2), (0, 1)
            )
            if self._first_step == epoch or not use_reference:
                input_pcd = to_sum_fmt(
                    pointcloud[:max_outputs, ::pcd_step, :3], (0, 0), torch.float32
                )
                if save_gt:
                    gtl = input_data.get("label", input_data.get("labels", None))
                    if gtl is None:
                        raise ValueError("input_data does not have label(s).")
                    gt_labels = to_sum_fmt(gtl[:max_outputs, ::pcd_step], (0, 1))
        else:
            raise NotImplementedError(
                "Saving 3D summary for the model "
                f"{self.model.cfg['name']} is not implemented."
            )

        def get_reference_or(data_tensor):
            if self._first_step == epoch or not use_reference:
                return data_tensor
            return self._first_step

        summary_dict = {
            "semantic_segmentation": {
                "vertex_positions": get_reference_or(input_pcd),
                "vertex_gt_labels": get_reference_or(gt_labels),
                "vertex_predict_labels": predict_labels,
                "label_to_names": label_to_names,
            }
        }
        return summary_dict

    def save_logs(self, writer, epoch):
        """Save logs from the training and send results to TensorBoard."""
        train_accs = self.metric_train.acc()

        train_ious = self.metric_train.iou()
        loss_dict = {
            "Training loss": np.mean(self.losses),
            "Validation loss": np.mean(self.valid_losses),
        }

        acc_dicts = [{"Training accuracy": acc} for acc in train_accs]

        iou_dicts = [{"Training IoU": iou} for iou in train_ious]

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
        log.info(f"Mean acc train: {acc_dicts[-1]['Training accuracy']:.3f} ")
        log.info(f"Mean IoU train: {iou_dicts[-1]['Training IoU']:.3f} ")

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

    def load_ckpt(
        self, ckpt_path=None, is_resume=True, from_binary=False, freeze_encoder=False
    ):
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
        if from_binary:
            ckpt["model_state_dict"].pop("fc1.3.conv.bias")
            ckpt["model_state_dict"].pop("fc1.3.conv.weight")

        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if freeze_encoder:
            for name, params in self.model.named_parameters():
                if name.startswith(("decoder", "fc1")):
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        if not from_binary:
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
                model_state_dict=self.student_model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
            ),
            join(path_ckpt, f"student_ckpt_{epoch:05d}.pth"),
        )
        torch.save(
            dict(
                epoch=epoch,
                model_state_dict=self.teacher_model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
            ),
            join(path_ckpt, f"teacher_ckpt_{epoch:05d}.pth"),
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


PIPELINE._register_module(SemiSupervisedSegmentation, "torch")
