# Standard Library
import os
import typing as t
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import numpy as np
import yaml
from easydict import EasyDict as edict


@dataclass
class OptionsDatasetShapenet:
    num_points: int
    resize_with_constant_border: bool


@dataclass
class OptionsDatasetPredict:
    folder: str


@dataclass
class OptionsDataset:
    name = "shapenet"
    subset_train = "train_small"
    subset_eval = "test_small"
    camera_f = [248.0, 248.0]
    camera_c = [111.5, 111.5]
    mesh_pos = [0.0, 0.0, -0.8]
    normalization = True
    num_classes = 13

    shapenet: OptionsDatasetShapenet
    predict: OptionsDatasetPredict


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
class OptionsTrian:
    num_epochs: int
    checkpoint_steps: int
    test_epochs: int
    shuffle: bool


@dataclass
class OptionsTest:
    shuffle: bool
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
    version: str | None
    num_workers: int
    num_gpus: int

    log_dir: str
    log_level: str
    checkpoint: str | None  # Checkpointへのpath
    dataset: OptionsDataset
    model: OptionsModel
    loss: OptionsLoss
    batch_size: int
    train: OptionsTrian
    test: OptionsTest
    optim: OptionsOptim

    mtl_filepath: str
    usemtl_name: str


options = t.cast(Options, edict())

options.name = "p2m"
options.version = None
options.num_workers = 1
options.num_gpus = 1

options.log_dir = "logs"
options.log_level = "info"
options.checkpoint = None
options.batch_size = 4
options.mtl_filepath = "rendering.mtl"
options.usemtl_name = "Default_OBJ.001"

options.dataset = edict()
options.dataset.name = "shapenet"
options.dataset.subset_train = "train_small"
options.dataset.subset_eval = "test_small"
options.dataset.camera_f = [248.0, 248.0]
options.dataset.camera_c = [111.5, 111.5]
options.dataset.mesh_pos = [0.0, 0.0, -0.8]
options.dataset.normalization = True
options.dataset.num_classes = 13

options.dataset.shapenet = edict()
options.dataset.shapenet.num_points = 3000
options.dataset.shapenet.resize_with_constant_border = False

options.dataset.predict = edict()
options.dataset.predict.folder = "/tmp"

options.model = edict()
options.model.name = "pixel2mesh"
options.model.hidden_dim = 192
options.model.last_hidden_dim = 192
options.model.coord_dim = 3
options.model.backbone = "vgg16"
options.model.gconv_activation = True
# provide a boundary for z, so that z will never be equal to 0, on denominator
# if z is greater than 0, it will never be less than z;
# if z is less than 0, it will never be greater than z.
options.model.z_threshold = 0
# align with original tensorflow model
# please follow experiments/tensorflow.yml
options.model.align_with_tensorflow = False

options.loss = edict()
options.loss.weights = edict()
options.loss.weights.normal = 1.6e-4
options.loss.weights.edge = 0.3
options.loss.weights.laplace = 0.5
options.loss.weights.move = 0.1
options.loss.weights.constant = 1.0
options.loss.weights.chamfer = [1.0, 1.0, 1.0]
options.loss.weights.chamfer_opposite = 1.0
options.loss.weights.reconst = 0.0

options.train = edict()
options.train.num_epochs = 50
options.train.checkpoint_steps = 10000
options.train.test_epochs = 1
options.train.shuffle = True

options.test = edict()
options.test.dataset = []
options.test.shuffle = False
options.test.weighted_mean = False

options.optim = edict()
options.optim.name = "adam"
options.optim.adam_beta1 = 0.9
options.optim.sgd_momentum = 0.9
options.optim.lr = 5.0e-5
options.optim.wd = 1.0e-6
options.optim.lr_step = [30, 45]
options.optim.lr_factor = 0.1


def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def _update_options(options_file: Path):
    # do scan twice
    # in the first round, MODEL.NAME is located so that we can initialize MODEL.EXTRA
    # in the second round, we update everything

    with open(options_file, mode="rt") as f:
        options_dict = yaml.safe_load(f)
        # do a dfs on `BASED_ON` options files
        if "based_on" in options_dict:
            for base_options in options_dict["based_on"]:
                _update_options(options_file.parent / base_options)
            options_dict.pop("based_on")
        _update_dict("", options_dict, options)


def update_options(options_file: Path):
    _update_options(options_file)


def gen_options(options_file):
    def to_dict(ed):
        ret = dict(ed)
        for k, v in ret.items():
            if isinstance(v, edict):
                ret[k] = to_dict(v)
            elif isinstance(v, np.ndarray):
                ret[k] = v.tolist()
        return ret

    cfg = to_dict(options)

    with open(options_file, "w") as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False)


def slugify(filename):
    filename = os.path.relpath(filename, ".")
    if filename.startswith("experiments/"):
        filename = filename[len("experiments/") :]
    return os.path.splitext(filename)[0].lower().replace("/", "_").replace(".", "_")


def reset_options(options, args) -> None:

    options.name = args.name


if __name__ == "__main__":
    parser = ArgumentParser("Read options and freeze")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    update_options(args.input)
    gen_options(args.output)
