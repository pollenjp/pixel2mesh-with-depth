# Standard Library
import typing as t
from enum import Enum
from enum import unique

# Third Party Library
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party Library
from p2m.datasets.shapenet_with_depth import P2MWithDepthBatchData
from p2m.models.backbones import get_backbone
from p2m.models.layers.gbottleneck import GBottleneck
from p2m.models.layers.gconv import GConv
from p2m.models.layers.gpooling import GUnpooling
from p2m.models.layers.gprojection import GProjection


@unique
class MergeType(Enum):
    CONCAT_AND_REDUCTION = "concat_and_reduction"
    ADD = "add"


class MergeFeatures(nn.Module):
    def __init__(self, merge_type: MergeType, options: t.Any | None = None):
        super().__init__()

        self.merge_type = merge_type

        match self.merge_type:
            case MergeType.CONCAT_AND_REDUCTION:
                if options is None:
                    raise ValueError("options must be provided for concat_and_reduction")

                in_dims: list[int] = options["in_dims"]
                out_dims: list[int] = options["out_dims"]
                if len(in_dims) != len(out_dims):
                    raise ValueError(
                        f"in_dims and out_dims must have the same length: {len(in_dims)} != {len(out_dims)}"
                    )
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Conv2d(in_dims[i], out_dims[i], kernel_size=1) for i in range(len(in_dims))]
                )

    def forward(self, a_feature: t.Sequence[torch.Tensor], b_feature: t.Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """_summary_

        Args:
            a_feature (t.Sequence[torch.Tensor]):
                tensor size: (batch_size, c, h, w)
            b_feature (t.Sequence[torch.Tensor]):
                tensor size: (batch_size, c, h, w)

        Raises:
            NotImplementedError: _description_

        Returns:
            list[torch.Tensor]: _description_
        """
        vals = []
        match self.merge_type:
            case MergeType.CONCAT_AND_REDUCTION:
                # concatenate along the channel dimension and reduce with a 1x1 conv
                for i, (a, b) in enumerate(zip(a_feature, b_feature)):
                    x = torch.cat([a, b], dim=1)
                    vals.append(self.layers[i](x))

            case MergeType.ADD:
                for a, b in zip(a_feature, b_feature):
                    vals.append(a + b)
            case _:
                raise NotImplementedError(f"Merge type {self.merge_type} not implemented")
        return vals


class P2MModelWithDepth(nn.Module):
    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super().__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.parameter.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.depth_nn_encoder, self.depth_nn_decoder = get_backbone(options)

        # self.merge_features = MergeFeatures(MergeType.ADD)
        block_dims = [
            256,
            512,
            1024,
            2048,
        ]
        self.merge_features = MergeFeatures(
            MergeType.CONCAT_AND_REDUCTION,
            options={
                "in_dims": [d * 2 for d in block_dims],
                "out_dims": block_dims,
            },
        )

        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList(
            [
                GBottleneck(
                    6,
                    self.features_dim,
                    self.hidden_dim,
                    self.coord_dim,
                    ellipsoid.adj_mat[0],
                    activation=self.gconv_activation,
                ),
                GBottleneck(
                    6,
                    self.features_dim + self.hidden_dim,
                    self.hidden_dim,
                    self.coord_dim,
                    ellipsoid.adj_mat[1],
                    activation=self.gconv_activation,
                ),
                GBottleneck(
                    6,
                    self.features_dim + self.hidden_dim,
                    self.hidden_dim,
                    self.last_hidden_dim,
                    ellipsoid.adj_mat[2],
                    activation=self.gconv_activation,
                ),
            ]
        )

        self.unpooling = nn.ModuleList([GUnpooling(ellipsoid.unpool_idx[0]), GUnpooling(ellipsoid.unpool_idx[1])])

        self.projection = GProjection(
            mesh_pos, camera_f, camera_c, bound=options.z_threshold, tensorflow_compatible=options.align_with_tensorflow
        )

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim, adj_mat=ellipsoid.adj_mat[2])

    def forward(self, batch: P2MWithDepthBatchData):
        img = batch["images"]
        batch_size = img.size(0)
        img_features = self.nn_encoder(img)
        depth_img_feats = self.depth_nn_encoder(batch["depth_images"].repeat(1, 3, 1, 1))

        encoded_features = self.merge_features(img_features, depth_img_feats)

        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, encoded_features, init_pts)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, encoded_features, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, encoded_features, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(encoded_features)
        else:
            reconst = None

        return {"pred_coord": [x1, x2, x3], "pred_coord_before_deform": [init_pts, x1_up, x2_up], "reconst": reconst}


class P2MModelWithDepthForwardReturn(t.TypedDict):
    pred_coord: list[torch.Tensor]  # (3,) arary. Each element is (batch_size, num_points, 3)
    pred_coord_before_deform: list[torch.Tensor]  # [init_pts, x1_up, x2_up]
    reconst: torch.Tensor  # TODO: ???
