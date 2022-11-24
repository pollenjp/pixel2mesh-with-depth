# Standard Library
from logging import getLogger

# Third Party Library
import numpy as np
import numpy.typing as npt
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import TensorBoardLogger

logger = getLogger(__name__)


def pl_log_scalar(pl_logger: Logger, tag: str, scalar: float | int, global_step: int) -> None:

    if isinstance(pl_logger, TensorBoardLogger):
        # > add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
        # > <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_images>
        pl_logger.experiment.add_scalar(tag=tag, scalar_value=scalar, global_step=global_step)
    else:
        error_msg: str = f"{pl_logger} doesn't support scalar logging"
        logger.error(error_msg)


def pl_log_images(pl_logger: Logger, tag: str, imgs_arr: npt.NDArray[np.uint8], global_step: int) -> None:
    if isinstance(pl_logger, TensorBoardLogger):
        logger.debug(f"pl_logger = {pl_logger}")
        logger.debug(f"imgs_arr.shape = {imgs_arr.shape}")
        # > add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
        # > <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_images>
        pl_logger.experiment.add_images(
            tag=tag, img_tensor=imgs_arr, global_step=global_step, walltime=None, dataformats="NHWC"
        )
    else:
        error_msg: str = f"{pl_logger} dones't support image logging"
        logger.error(error_msg)
