# Standard Library
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class OptionsDatasetShapenet:
    num_points: int
    resize_with_constant_border: bool


@dataclass
class OptionsDatasetPredict:
    folder: str


@dataclass
class OptionsDataset:
    name: str
    num_classes: int
    subset_train: str
    subset_eval: str
    camera_f: list[float]
    camera_c: list[float]
    mesh_pos: list[float]
    shapenet: OptionsDatasetShapenet
    predict: OptionsDatasetPredict

    normalization: bool = True


@dataclass
class OptionsModel:
    name: str
    hidden_dim: int
    last_hidden_dim: int
    coord_dim: int
    backbone: str
    gconv_activation: bool

    # provide a boundary for z, so that z will never be equal to 0, on denominator
    # if z is greater than 0, it will never be less than z;
    # if z is less than 0, it will never be greater than z.
    z_threshold: int

    # align with original tensorflow model
    # please follow experiments/tensorflow.yml
    align_with_tensorflow: bool


@dataclass
class OptionsLossWeights:
    normal: float
    edge: float
    laplace: float
    move: float
    constant: float
    chamfer: list[float]
    chamfer_opposite: float
    reconst: float


@dataclass
class OptionsLoss:
    weights: OptionsLossWeights


@dataclass
class OptionsTest:
    weighted_mean: bool


@dataclass
class OptionsOptim:
    name: str
    adam_beta1: float
    sgd_momentum: float
    lr: float
    wd: float
    lr_step: list[int]  # 2 elements
    lr_factor: float


@dataclass
class Options:
    name: str
    num_workers: int
    num_gpus: int

    log_root_path: str
    dataset_root_path: str
    checkpoint_path: str | None  # Checkpointへのpath
    dataset: OptionsDataset
    model: OptionsModel
    loss: OptionsLoss
    batch_size: int
    test: OptionsTest
    optim: OptionsOptim
    num_epochs: int
    random_seed: int

    mtl_filepath: str
    usemtl_name: str


def assert_sequence_config(cfg: MutableSequence) -> None:
    for val in cfg:
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)
        else:
            print(f"{val}")


def assert_mapping_config(cfg: MutableMapping) -> None:
    for _, val in cfg.items():
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)
