# Standard Library
import typing as t

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.chamfer import knn_points

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


def calc_f1_score(
    dis_to_pred: torch.Tensor,
    dis_to_gt: torch.Tensor,
    pred_length: int,
    gt_length: int,
    thresh: float,
) -> torch.Tensor:
    recall = (dis_to_gt < thresh).sum() / gt_length
    precision = (dis_to_pred < thresh).sum() / pred_length
    return 2 * precision * recall / (precision + recall + 1e-8)


def plot_point_cloud(vertices: t.Sequence[torch.Tensor] | torch.Tensor, num_cols: int = 2) -> npt.NDArray[np.uint8]:
    """_summary_

    Args:
        vertices (torch.Tensor):
            Size (num_batch_size, num_vertices, 3)

    Returns:
        npt.NDArray[np.uint8]: _description_
    """

    num_plot = len(vertices)
    num_rows = (num_plot // num_cols) + (0 if num_plot % num_cols == 0 else 1)
    figure_size_unit = 2

    fig = plt.figure(
        facecolor="white",
        figsize=(figure_size_unit * num_cols, figure_size_unit * num_rows),
        tight_layout=True,
    )

    for i, vs in enumerate(vertices):
        # vs: (num_vertices, 3)
        x, y, z = vs.detach().cpu().squeeze().unbind(1)
        ax = t.cast(Axes3D, fig.add_subplot(num_rows, num_cols, i + 1, projection="3d"))
        ax.scatter3D(x, y, z, s=0.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # 右手系にする
        ax.invert_xaxis()
        ax.invert_yaxis()
        view_elev, view_azim = 20, 135
        ax.view_init(elev=view_elev, azim=view_azim)

    # fig.subplots_adjust(wspace=0.5, hspace=0.3)
    space = 5.0
    fig.subplots_adjust(wspace=space, hspace=space)
    fig.canvas.draw()

    img: npt.NDArray = np.frombuffer(buffer=fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img


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
        batch_size: int = len(batch["images"])

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

        self.log(name="train_loss", value=loss, batch_size=batch_size)

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

        batch_size: int = len(batch["images"])

        if batch_idx == 0:
            self.validation_epoch_start(phase_name=phase_name)

        preds: P2MModelWithTemplateForwardReturn = self(batch)
        summary: P2MLossForwardReturnSecondDict
        _, summary = self.p2m_loss(preds, batch)

        # for monitor label
        self.log(name=f"{phase_name}_loss", value=summary["loss"], batch_size=batch_size)

        for loss_name, avg_meter in self.val_epoch_loss_avg_meters.items():
            avg_meter.update(summary[loss_name])
            self.log(name=f"{phase_name}/{loss_name}", value=summary[loss_name], batch_size=batch_size)

        # calculate evaluation metrics
        pred_vertices = preds["pred_coord"][-1]  # (batch_size, num_vertices, 3)
        gt_points = batch["points_orig"]  # (batch_size, ) array. torch.Tensor(num_points, 3)

        total_chamfer_distance = 0.0
        total_f1_tau = 0.0
        total_f1_2tau = 0.0

        for i, (label, pred_v, gt_v) in enumerate(zip(batch["labels"], pred_vertices, gt_points)):
            pred_v = pred_v.unsqueeze(0)  # (1, num_vertices, 3)
            gt_v = gt_v.unsqueeze(0)

            # chamfer distance
            cd = t.cast(
                torch.Tensor,
                chamfer_distance(pred_v, gt_v)[0],
            )
            self.log(name=f"{phase_name}_eval_cd/{label}", value=cd, batch_size=1)
            total_chamfer_distance += cd

            # f1 score
            knn1 = knn_points(pred_v, gt_v, K=1)
            knn2 = knn_points(gt_v, pred_v, K=1)
            dis_to_pred = knn1.dists.squeeze(0).squeeze(1)  # (num_gt_points, )
            dis_to_gt = knn2.dists.squeeze(0).squeeze(1)
            pred_length = dis_to_pred.size(0)
            gt_length = dis_to_gt.size(0)
            tau: float = 1e-4
            f1_tau = calc_f1_score(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh=tau)
            f1_2tau = calc_f1_score(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh=2 * tau)

            self.log(name=f"{phase_name}_eval_f1_tau/{label}", value=f1_tau, batch_size=1)
            self.log(name=f"{phase_name}_eval_f1_2tau/{label}", value=f1_2tau, batch_size=1)
            total_f1_tau += f1_tau
            total_f1_2tau += f1_2tau

        self.log(name=f"{phase_name}_eval_cd", value=total_chamfer_distance / batch_size, batch_size=batch_size)
        self.log(name=f"{phase_name}_eval_f1_tau", value=total_f1_tau / batch_size, batch_size=batch_size)
        self.log(name=f"{phase_name}_eval_f1_2tau", value=total_f1_2tau / batch_size, batch_size=batch_size)

        if batch_idx == 0:
            self.log_image(
                tag=f"{phase_name}_imgs",
                imgs_arr=batch["images_orig"],
                global_step=self.val_global_step,
            )

            self.log_image(
                tag=f"{phase_name}_vertices/gt_pred",
                imgs_arr=np.array(
                    [
                        plot_point_cloud(vertices=gt_points, num_cols=1),
                        plot_point_cloud(vertices=batch["init_pts"], num_cols=1),
                        plot_point_cloud(vertices=pred_vertices, num_cols=1),
                    ]
                ),
                global_step=self.val_global_step,
            )

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
        imgs_arr: torch.Tensor | npt.NDArray[np.uint8],
        global_step: int,
    ) -> None:

        for pl_logger in self.loggers:
            pl_loggers.pl_log_images(
                pl_logger=pl_logger,
                tag=tag,
                imgs_arr=imgs_arr,
                global_step=global_step,
            )
