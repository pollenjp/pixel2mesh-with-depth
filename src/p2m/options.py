# Standard Library
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from enum import Enum
from enum import unique
from logging import NullHandler
from logging import getLogger

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class OptionsDatasetShapenet:
    num_points: int
    resize_with_constant_border: bool


@dataclass
class OptionsDataset:
    name: str
    num_classes: int
    label_json_path: str
    root_path: str
    train_list_filepath: str
    val_list_filepath: str
    test_list_filepath: str | None
    camera_f: list[float]
    camera_c: list[float]
    mesh_pos: list[float]
    shapenet: OptionsDatasetShapenet

    normalization: bool = True


@unique
class ModelName(Enum):
    P2M = 0
    P2M_WITH_TEMPLATE = 1
    P2M_WITH_DEPTH = 2


@unique
class ModelBackbone(Enum):
    RESNET50 = 0
    VGG16 = 1


@dataclass
class OptionsModel:
    name: ModelName
    hidden_dim: int
    last_hidden_dim: int
    coord_dim: int
    backbone: ModelBackbone
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


@unique
class OptimName(Enum):
    ADAM = 0
    SGD = 1


@dataclass
class OptionsOptim:
    name: OptimName
    lr: float
    lr_step: list[int]  # 2 elements
    lr_factor: float
    weight_decay: float

    # adam:
    #   - Adam beta1
    #   - Adam beta2
    # sgd:
    #   - SGD momentum
    params: list[int | float]


@dataclass
class Options:
    num_workers: int

    datetime: str  # %Y-%m-%dT%H%M%S
    log_root_path: str
    checkpoint_path: str | None  # Checkpointへのpath
    dataset: OptionsDataset
    model: OptionsModel
    loss: OptionsLoss
    batch_size: int
    optim: OptionsOptim
    num_epochs: int
    random_seed: int

    batch_size_for_plot: int
    mtl_filepath: str
    usemtl_name: str


def assert_sequence_config(cfg: MutableSequence) -> None:
    for val in cfg:
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)


def assert_mapping_config(cfg: MutableMapping) -> None:
    for _, val in cfg.items():
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)
