# Standard Library
import typing as t

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D


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
