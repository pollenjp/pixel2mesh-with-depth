# Standard Library
import typing as t
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
from p2m import ModelName
from p2m import get_module_set
from p2m.options import Options
from p2m.options import assert_mapping_config
from p2m.utils.logger import reset_logging_config

logger = getLogger(__name__)
logger.addHandler(NullHandler())


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
        ModelName[f"{options.model.name}".upper()],
        options=options,
    )

    logger_root_path = Path(options.log_root_path) / "lightning_logs"
    logger.info(f"{logger_root_path=}")

    pl_loggers: list[Logger] = [
        TensorBoardLogger(save_dir=logger_root_path / "tensorboard", name=options.model.name.name)
    ]

    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=logger_root_path,
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
