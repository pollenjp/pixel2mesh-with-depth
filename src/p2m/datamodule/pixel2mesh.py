# Standard Library
from logging import NullHandler
from logging import getLogger

# Third Party Library
from torch.utils.data import DataLoader

# First Party Library
from p2m.datasets.shapenet import ShapeNet
from p2m.datasets.shapenet import get_shapenet_collate

# Local Library
from .base import DataModule

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class ShapeNetDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        logger.info("prepare_data")

    def prepare_data_per_node(self) -> None:
        logger.info("prepare_data_per_node")

    def setup(self, stage: str | None = None):
        if stage is not None:
            if stage not in ["fit", "test"]:
                raise ValueError(f"stage = {stage}")

    def train_dataloader(self):

        train_dataset = ShapeNet(
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
            collate_fn=get_shapenet_collate(self.options.dataset.shapenet.num_points),
        )

        return train_dataloader

    def val_dataloader(self):
        val_dataset = ShapeNet(
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
            collate_fn=get_shapenet_collate(self.options.dataset.shapenet.num_points),
        )

        return val_dataloader

    def test_dataloader(self):

        raise NotImplementedError("test_dataloader is not implemented!")
        test_dataset = ShapeNet(
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
