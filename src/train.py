# Standard Library
import datetime
import sys
import typing as t
from dataclasses import dataclass
from enum import Enum
from enum import unique
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import pytorch_lightning as pl
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
from p2m.options import options as opt
from p2m.options import reset_options
from p2m.options import update_options

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class Args:
    name: str
    option_path: Path


def get_args() -> Args:
    # Standard Library
    import argparse

    parser = argparse.ArgumentParser(description="Pixel2Mesh Training")

    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--option_path", required=True, type=lambda x: Path(x).expanduser().absolute())

    args = parser.parse_args()

    return Args(
        name=args.name,
        option_path=args.option_path,
    )


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
    dataset_root_path: Path,
    options: Options,
) -> tuple[DataModule, pl.LightningModule]:
    match model_name:
        case ModelName.P2M | ModelName.P2M_WITH_TEMPLATE:
            module_set = name2module[model_name]
            return (
                module_set.data_module(
                    data_root_path=dataset_root_path,
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


def main():
    # Standard Library
    from logging.config import dictConfig

    dictConfig(
        {
            "version": 1,
            "formatters": {
                "console_formatter": {
                    "format": "".join(
                        [
                            "[%(asctime)s]",
                            "[%(name)20s]",
                            "[%(levelname)10s]",
                            "[%(threadName)10s]",
                            "[%(processName)10s]",
                            "[%(filename)20s:%(lineno)4d]",
                            " - %(message)s",
                        ]
                    ),
                }
            },
            "handlers": {
                "console_handler": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "console_formatter",
                }
            },
            "disable_existing_loggers": True,
            "loggers": {
                "": {},  # disable the root logger
                "__main__": {
                    "level": "DEBUG",
                    "handlers": ["console_handler"],
                },
            },
        }
    )

    args = get_args()

    update_options(args.option_path)  # this updates options

    options = t.cast(Options, opt)

    dataset_root_path = Path("datasets") / "data" / "shapenet_with_template"
    random_seed: int = 0
    seed_everything(random_seed, workers=True)

    dm, model = get_module_set(
        ModelName[f"{args.name}".upper()],
        dataset_root_path=dataset_root_path,
        options=options,
    )

    logger_root_path = (
        Path(options.log_dir)
        / "lightning_logs"
        / f"{options.name}"
        / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    logger.info(f"{logger_root_path=}")

    pl_loggers: list[Logger] = [TensorBoardLogger(save_dir=logger_root_path / "tensorboard", name=options.name)]

    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=logger_root_path,
        # gpus=cfg.pytorch_lightning.gpus,
        logger=pl_loggers,
        max_epochs=options.train.num_epochs,
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
        resume_from_checkpoint=options.checkpoint,
        accelerator="gpu",
        devices=1,
        deterministic="warn",
    )

    # fit
    logger.info("Begin fit!")
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
