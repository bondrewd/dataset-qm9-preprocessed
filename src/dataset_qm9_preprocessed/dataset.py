import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import requests
import torch
from torch.utils.data import Dataset

from dataset_qm9_preprocessed.utils import data_dict_from_xyz_str

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent


class QM9Dataset(Dataset):
    def __init__(self, url: Optional[str] = None, dataset_dir_path: Optional[str] = None):
        if url is None:
            self.url = "https://github.com/bondrewd/dataset-qm9-raw/raw/refs/heads/main/dsgdb9nsd.xyz.tar.bz2"
        else:
            self.url = url

        if dataset_dir_path is None:
            self.dataset_dir_path = PROJECT_ROOT_PATH / Path("../../../dataset")
        else:
            self.dataset_dir_path = Path(dataset_dir_path)

        self.dataset_data_path = Path(self.dataset_dir_path / "dataset-qm9").with_suffix(".pth")

        if not self.dataset_data_path.exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Step 1: convert temp dir to a Path
                tmp_dir_path = Path(tmp_dir)

                # Step 2: download raw data
                raw_data_path = tmp_dir_path / "data.tar.bz2"
                if not raw_data_path.exists():
                    response = requests.get(self.url)
                    if response.status_code == 200:
                        with open(raw_data_path, "wb") as file:
                            file.write(response.content)
                    else:
                        raise RuntimeError(f"Failed to download raw data from {self.url}")

                # Step 3: decompress raw data
                raw_data_dir_path = tmp_dir_path / "raw"
                if not raw_data_dir_path.exists():
                    raw_data_dir_path.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(raw_data_path, "r:bz2") as tar:
                        tar.extractall(path=raw_data_dir_path, filter="fully_trusted")

                # Step 4: list .xyz files
                xyz_file_paths = list(raw_data_dir_path.rglob("*.xyz"))
                xyz_file_paths.sort()

                # Step 5: build data dictionaries
                self.data_dicts = []
                for xyz_file_path in xyz_file_paths:
                    try:
                        with open(xyz_file_path, "r") as xyz_file:
                            xyz_str = xyz_file.read()
                        data_dict = data_dict_from_xyz_str(xyz_str)
                        data_dict["x"] = data_dict["x"] - torch.mean(data_dict["x"], dim=0, keepdim=True)
                        if data_dict["x_ctx"] is not None:
                            data_dict["x_ctx"] = data_dict["x_ctx"] - torch.mean(data_dict["x_ctx"], dim=0, keepdim=True)
                        self.data_dicts.append(data_dict)
                    except Exception as e:
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise
                        continue

                # Step 6: save data dictionaries
                self.dataset_dir_path.mkdir(parents=True, exist_ok=True)
                torch.save(self.data_dicts, self.dataset_data_path)
        else:
            self.data_dicts = torch.load(self.dataset_data_path, weights_only=True)

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        return self.data_dicts[idx]


if __name__ == "__main__":
    dataset = QM9Dataset()
    print("QM9 Dataset")
    print(f"Length: {len(dataset)}")
    print(f"Location: {dataset.dataset_data_path.resolve()}")
