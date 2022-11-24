# Standard Library
import typing as t
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# First Party Library
from p2m.datasets.shapenet_with_template import ShapeNetWithTemplate

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class ShapeNetWithTemplateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_path: Path,
        options: t.Any,
        *,
        batch_size: int,
        num_workers: int = 4,
        random_seed: int | None = None,
    ):
        """(override)"""
        super().__init__()

        self.data_root_path = data_root_path
        self.options = options
        self.batch_size: int = batch_size
        self.num_workers = num_workers
        self.random_seed: int | None = random_seed

    def prepare_data(self):

        logger.info("prepare_data")

    def setup(self, stage: str | None = None):
        if stage is not None:
            if stage not in ["fit", "test"]:
                raise ValueError(f"stage = {stage}")

    def train_dataloader(self):

        train_dataset = ShapeNetWithTemplate(
            file_root=self.data_root_path,
            file_list_name=self.options.dataset.subset_train,
            mesh_pos=self.options.dataset.mesh_pos,
            normalization=self.options.dataset.normalization,
            shapenet_options=self.options.dataset.shapenet,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            # collate_fn=collate_fn,
        )

        return train_dataloader

    def val_dataloader(self):
        val_dataset = ShapeNetWithTemplate(
            file_root=self.data_root_path,
            file_list_name=self.options.dataset.subset_eval,
            mesh_pos=self.options.dataset.mesh_pos,
            normalization=self.options.dataset.normalization,
            shapenet_options=self.options.dataset.shapenet,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=collate_fn,
        )

        return val_dataloader

    def test_dataloader(self):

        raise NotImplementedError("test_dataloader is not implemented!")
        test_dataset = ShapeNetWithTemplate(
            file_root=self.data_root_path,
            file_list_name=self.options.dataset.subset_eval,
            mesh_pos=self.options.dataset.mesh_pos,
            normalization=self.options.dataset.normalization,
            shapenet_options=self.options.dataset.shapenet,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            shuffle=False,
            # drop_last=True,
            # collate_fn=collate_fn,
        )

        return test_dataloader

    # def transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None):
    #     """move all tensors in your custom data structure to the device
    #     * <https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#transfer-batch-to-device>
    #     * <https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#transfer-batch-to-device>
    #     """
    #     logger.debug(f"{inspect.getframeinfo(inspect.currentframe()).function} - start (device={device})")

    #     if not isinstance(batch, ShapeNetBatchDataPair):
    #         # batch = super().transfer_batch_to_device(batch, device)
    #         raise ValueError(f"{type(batch)} is not supported!")

    #     for member_key, member_val in batch.__dict__.items():
    #         if torch.is_tensor(member_val):
    #             member_val: torch.Tensor
    #             batch.__dict__[member_key] = member_val.to(device)
    #     return batch
