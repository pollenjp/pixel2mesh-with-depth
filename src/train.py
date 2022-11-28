# Standard Library
import argparse
import datetime
import sys
import typing as t
from pathlib import Path

# Third Party Library
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import TensorBoardLogger

# First Party Library
from p2m.datamodule.pixel2mesh_with_template import ShapeNetWithTemplateDataModule
from p2m.lightningmodule.pixel2mesh_with_template import P2MModelWithTemplateModule
from p2m.options import Options
from p2m.options import options as opt
from p2m.options import reset_options
from p2m.options import update_options


def parse_args():
    parser = argparse.ArgumentParser(description="Pixel2Mesh Training Entrypoint")
    parser.add_argument("--options", help="experiment options file name", required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument("--batch-size", help="batch size", type=int)
    parser.add_argument("--checkpoint", help="checkpoint file", type=str)
    parser.add_argument("--num-epochs", help="number of epochs", type=int)
    parser.add_argument("--version", help="version of task (timestamp by default)", type=str)
    parser.add_argument("--name", required=True, type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(opt, args)

    options = t.cast(Options, opt)

    dataset_root_path = Path("datasets") / "data" / "shapenet_with_template"
    random_seed: int = 0

    dm = ShapeNetWithTemplateDataModule(
        data_root_path=dataset_root_path,
        options=options,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        random_seed=random_seed,
    )

    model = P2MModelWithTemplateModule(options=options)

    logger_root_path = (
        Path("logs") / "lightning_logs" / f"{options.name}" / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    pl_loggers: list[Logger] = [TensorBoardLogger(save_dir=logger_root_path / "tensorboard", name=options.name)]

    ckpt_path: Path | None = None

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
        resume_from_checkpoint=ckpt_path,
        accelerator="gpu",
        devices=1,
    )

    # fit
    logger.info("Begin fit!")
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
