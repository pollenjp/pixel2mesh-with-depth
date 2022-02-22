import logging
import pickle
from logging import getLogger
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
    level=logging.WARNING,
)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    pkl_path: Path = Path("/media/data_hdd/hiroakisugisaki/data/dataset/ShapeNet_for_P2M/ShapeNetP2M/02691156/10155655850468db78d106ce0a280f87/rendering/01.dat")
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path}")

    with open(pkl_path, mode="rb") as f:
        data = pickle.load(f, encoding="latin1")

    logger.info(f"type(data)={type(data)}")
    logger.info(f"data.shape={data.shape}")

    pts, normals = data[:, :3], data[:, 3:]

    logger.info(f"type(pts)={type(pts)}")
    logger.info(f"pts.shape={pts.shape}")

    logger.info(f"type(normals)={type(normals)}")
    logger.info(f"normals.shape={normals.shape}")
