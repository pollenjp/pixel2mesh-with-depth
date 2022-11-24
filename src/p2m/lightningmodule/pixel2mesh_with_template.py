# Third Party Library
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch

# First Party Library
from p2m.datasets.shapenet_with_template import P2MWithTemplateBatchData
from p2m.models.losses.p2m import P2MLoss
from p2m.models.losses.p2m import P2MLossForwardReturnSecondDict
from p2m.models.p2m_with_template import P2MModelWithTemplate
from p2m.models.p2m_with_template import P2MModelWithTemplateForwardReturn
from p2m.options import Options
from p2m.utils import pl_loggers
from p2m.utils.average_meter import AverageMeter
from p2m.utils.mesh import Ellipsoid


class P2MModelWithTemplateModule(pl.LightningModule):
    def __init__(
        self,
        options: Options,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.options = options
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.model = P2MModelWithTemplate(
            self.options.model,
            self.ellipsoid,
            self.options.dataset.camera_f,
            self.options.dataset.camera_c,
            self.options.dataset.mesh_pos,
        )
        self.p2m_loss = P2MLoss(self.options.loss, self.ellipsoid).cuda()

        self.train_epoch_loss_avg_meters = {
            "loss": AverageMeter(),
            "loss_chamfer": AverageMeter(),
            "loss_edge": AverageMeter(),
            "loss_laplace": AverageMeter(),
            "loss_move": AverageMeter(),
            "loss_normal": AverageMeter(),
        }
        self.val_epoch_loss_avg_meters = {
            "loss": AverageMeter(),
            "loss_chamfer": AverageMeter(),
            "loss_edge": AverageMeter(),
            "loss_laplace": AverageMeter(),
            "loss_move": AverageMeter(),
            "loss_normal": AverageMeter(),
        }

        self.val_global_step: int = 0

    def forward(self, batch: P2MWithTemplateBatchData) -> P2MModelWithTemplateForwardReturn:
        x: P2MModelWithTemplateForwardReturn = self.model(batch)
        return x

    def training_epoch_start(self, *, phase_name: str = "train") -> None:
        """
        custom method
        """
        return

    def training_step(
        self,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        # optimizer_idx: int,
        *,
        phase_name: str = "train",
    ) -> None:
        if batch_idx == 0:
            self.training_epoch_start(phase_name=phase_name)

        preds: P2MModelWithTemplateForwardReturn = self(batch)
        summary: P2MLossForwardReturnSecondDict
        loss, summary = self.p2m_loss(preds, batch)

        # Gradient decent
        self.model.zero_grad()

        # backward
        loss.backward()

        # step
        self.solver.step()

        self.log(name="train_loss", value=loss)

        for loss_name, avg_meter in self.train_epoch_loss_avg_meters.items():
            avg_meter.update(summary[loss_name])

        if batch_idx == 0:
            # TODO:
            pass

    def training_epoch_end(self, training_step_outputs, *, phase_name: str = "train") -> None:
        for loss_name, avg_meter in self.train_epoch_loss_avg_meters.items():
            self.log_scalar(
                tag=f"{phase_name}/epoch_{loss_name}",
                scalar=avg_meter.avg,
                global_step=self.current_epoch,
            )

    def validation_epoch_start(self, *, phase_name: str = "val"):

        return

    def validation_step(
        self,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        *,
        phase_name: str = "val",
    ):
        self.val_global_step += 1

        if batch_idx == 0:
            self.validation_epoch_start(phase_name=phase_name)

        preds: P2MModelWithTemplateForwardReturn = self(batch)
        summary: P2MLossForwardReturnSecondDict
        _, summary = self.p2m_loss(preds, batch)

        self.log(name=f"{phase_name}_loss", value=summary["loss"])

        for loss_name, avg_meter in self.val_epoch_loss_avg_meters.items():
            avg_meter.update(summary[loss_name])

        if batch_idx == 0:

            pass

        return

    def validation_epoch_end(
        self,
        validation_step_outputs,
        *,
        phase_name: str = "val",
    ) -> None:
        for loss_name, avg_meter in self.val_epoch_loss_avg_meters.items():
            self.log_scalar(
                tag=f"{phase_name}/epoch_{loss_name}",
                scalar=avg_meter.avg,
                global_step=self.current_epoch,
            )

    def test_epoch_start(self, *, phase_name: str = "test") -> None:
        """
        custom method
        """
        return

    def test_step(
        self,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        *,
        phase_name: str = "test",
    ):
        if batch_idx == 0:
            self.test_epoch_start(phase_name=phase_name)

        self.validation_step(batch=batch, batch_idx=batch_idx, phase_name=phase_name)
        return

    def test_eopch_end(self, test_step_outputs, *, phase_name: str = "test"):
        self.validation_epoch_end(validation_step_outputs=test_step_outputs, phase_name="test")
        return

    def configure_optimizers(self):

        # Set up solver
        self.solver: torch.optim.optimizer.Optimizer
        if self.options.optim.name == "adam":
            self.solver = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd,
            )
        elif self.options.optim.name == "sgd":
            self.solver = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd,
            )
        else:
            raise NotImplementedError(f"Your optimizer is not found: {self.options.optim.name}")

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.solver,
            milestones=self.options.optim.lr_step,
            gamma=self.options.optim.lr_factor,
        )

        optimizers: list[torch.optim.optimizer.Optimizer] = [
            self.solver,
        ]
        lr_scheculers: list = [
            self.lr_scheduler,
        ]

        return optimizers, lr_scheculers

    def log_scalar(
        self,
        tag: str,
        scalar: float | int,
        global_step: int,
    ) -> None:

        for pl_logger in self.loggers:
            pl_loggers.pl_log_scalar(
                pl_logger=pl_logger,
                tag=tag,
                scalar=scalar,
                global_step=global_step,
            )

    def log_image(
        self,
        tag: str,
        imgs_arr: npt.NDArray[np.uint8],
        global_step: int,
    ) -> None:

        for pl_logger in self.loggers:
            pl_loggers.pl_log_images(
                pl_logger=pl_logger,
                tag=tag,
                imgs_arr=imgs_arr,
                global_step=global_step,
            )
