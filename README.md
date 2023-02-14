# Pixel2Mesh With Depth

forked from <https://github.com/noahcao/Pixel2Mesh>

## Setup

p2m用のPython環境の構築

```sh
make setup
```

If download error occurs, see `setup.sh` and edit `pytorch3d_version_xyz` to the latest one.
The whl file URL of old version's PyTorch3D may be changed or removed.

## Example

### train

dataset file (`conf/dataset/*.yaml`) should be edited for your environment.

```sh
poetry run python src/train.py \
  model=pixel2mesh_with_depth_only_3d_cnn \
  dataset=resnet_with_template_airplane \
  model.backbone=RESNET50 \
  optim.lr=0.0001 \
  batch_size=32 \
  num_workers=16 \
  loss.weights.laplace=1.32
```

### test

```sh
(
ckpt_filename="val_loss=0.00099826-epoch=26-step=10908.ckpt";
poetry run python src/test.py \
  model=pixel2mesh_with_depth_3d_cnn \
  dataset=resnet_with_template_airplane \
  model.backbone=RESNET50 \
  batch_size=8 \
  num_workers=8 \
  "checkpoint_path=\"logs/train/P2M_WITH_DEPTH_3D_CNN/shapenet_with_template/2023-01-11T213742/lightning_logs/model-checkpoint/${ckpt_filename}\"";
)
```

### eval.py

推論したOBJファイルが格納されているディレクトリを `eval.py` 内の `dir_list` にハードコードする.
複数ディレクトリ指定可能であり, ディレクトリごとの結果を算出する.

```sh
poetry run python ./eval.py
```

### scripts/rust

Pixel2Meshのデータセットファイルの加工等.

[scripts/rust/README.md](./scripts/rust/README.md).

### scripts/rendering

生成したOBJファイルをblenderを経由してレンダリングする.
`scripts/rendering` 以下の[README.md](./scripts/rendering/README.md)を先に確認して環境構築されたし.
環境構築後は以下のコードで並列にレンダリングを行う.

```sh
(
base_dirpath="logs/test/P2M_WITH_DEPTH_3D_CNN/shapenet_with_template/2023-01-23T204537"
poetry run python \
  scripts/rendering/run_recursively.py \
  --data_dir "${base_dirpath}"/output \
  --out_dir "${base_dirpath}"/output_rendering \
  --num_workers 4
)
```
