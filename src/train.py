# Standard Library
import datetime
import typing as t
from dataclasses import dataclass
from enum import Enum
from enum import unique
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import TensorBoardLogger

# First Party Library
from p2m.datamodule.base import DataModule
from p2m.datamodule.pixel2mesh import ShapeNetDataModule
from p2m.datamodule.pixel2mesh_with_template import ShapeNetWithTemplateDataModule
from p2m.lightningmodule.pixel2mesh import P2MModelModule
from p2m.lightningmodule.pixel2mesh_with_template import P2MModelWithTemplateModule
from p2m.options import Options
from p2m.options import assert_mapping_config
from p2m.utils.logger import reset_logging_config

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@unique
class ModelName(Enum):
    P2M = "p2m"
    P2M_WITH_TEMPLATE = "p2m_with_template"


@dataclass
class ModuleSet:
    module: t.Type[pl.LightningModule]
    data_module: t.Type[DataModule]


name2module: dict[ModelName, ModuleSet] = {
    ModelName.P2M: ModuleSet(
        module=P2MModelModule,
        data_module=ShapeNetDataModule,
    ),
    ModelName.P2M_WITH_TEMPLATE: ModuleSet(
        module=P2MModelWithTemplateModule,
        data_module=ShapeNetWithTemplateDataModule,
    ),
}


def get_module_set(
    model_name: ModelName,
    options: Options,
) -> tuple[DataModule, pl.LightningModule]:
    match model_name:
        case ModelName.P2M | ModelName.P2M_WITH_TEMPLATE:
            module_set = name2module[model_name]
            return (
                module_set.data_module(
                    options=options,
                    batch_size=options.batch_size,
                    num_workers=options.num_workers,
                ),
                module_set.module(
                    options=options,
                ),
            )
        case _:
            raise ValueError(f"Unknown module name: {model_name}")


this_file_path = Path(__file__)
project_path = this_file_path.parents[1] / "conf"


@hydra.main(version_base=None, config_path=f"{project_path}", config_name=f"{this_file_path.stem}")
def main(cfg: DictConfig) -> None:
    cfg = t.cast(DictConfig, OmegaConf.merge(OmegaConf.structured(Options), cfg))
    assert_mapping_config(cfg)
    options = t.cast(Options, cfg)

    log_root_path = Path(options.log_root_path)
    reset_logging_config(log_root_path / "run.log")

    seed_everything(options.random_seed, workers=True)

    dm, model = get_module_set(
        ModelName[f"{options.name}".upper()],
        options=options,
    )

    logger_root_path = (
        Path(options.log_root_path) / "lightning_logs" / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    logger.info(f"{logger_root_path=}")

    pl_loggers: list[Logger] = [TensorBoardLogger(save_dir=logger_root_path / "tensorboard", name=options.name)]

    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=logger_root_path,
        # gpus=cfg.pytorch_lightning.gpus,
        logger=pl_loggers,
        max_epochs=options.num_epochs,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_last=True,
                save_top_k=5,
                dirpath=logger_root_path / "model-checkpoint",
                filename="{val_loss:.8f}-{epoch}-{step}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor="val_loss",
                patience=20,
            ),
        ],
        auto_select_gpus=True,
        resume_from_checkpoint=options.checkpoint_path,
        accelerator="gpu",
        devices=1,
        deterministic="warn",
    )

    # fit
    logger.info("Begin fit!")
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
