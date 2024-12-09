import argparse
import logging
import pathlib
import tarfile
import tomllib

import requests
import torch

from src.utils import data_dict_from_xyz_str


def main():
    # Setup logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Step 1: Parse arguments
    logging.info("Step 01 - Parse arguments")

    parser = argparse.ArgumentParser(description="Load a configuration file in TOML format.")
    parser.add_argument("cfg", type=str, help="Path to the configuration file in TOML format.")

    args = parser.parse_args()

    logging.info("Step 01 - Completed")

    # Step 2: Read configuration file
    logging.info("Step 02 - Read configuration file")

    cfg_path = pathlib.Path(args.cfg)
    if not cfg_path.exists() or not cfg_path.is_file():
        logging.error(f"Step 02 - The configuration file '{cfg_path}' does not exist or is not a valid file.")

    with cfg_path.open("rb") as toml_file:
        cfg = tomllib.load(toml_file)

    logging.info("Step 02 - Completed")

    # Step 3: Create temporary directory
    logging.info("Step 03 - Create temporary directory")

    tmp_dir_path = pathlib.Path("../tmp")
    tmp_dir_path.mkdir(parents=True, exist_ok=True)

    logging.info("Step 03 - Completed")

    # Step 4: Download raw data
    logging.info("Step 04 - Download raw data")

    bz2_path = tmp_dir_path / "data.tar.bz2"
    raw_data_url = cfg["url"]
    if not bz2_path.exists():  # delete
        response = requests.get(raw_data_url)
        if response.status_code == 200:
            with open(bz2_path, "wb") as file:
                file.write(response.content)
            logging.info(f"File downloaded: {bz2_path}")
        else:
            logging.error(f"Failed to download file: {response.status_code}")
            return

    logging.info(f"Step 04 - Raw data url: {raw_data_url}")
    logging.info(f"Step 04 - Raw data path: {bz2_path}")
    logging.info("Step 04 - Completed")

    # Step 5: Decompress raw data
    logging.info("Step 05 - Decompress raw data")

    raw_dir_path = tmp_dir_path / "raw"
    if not raw_dir_path.exists():  # delete
        raw_dir_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bz2_path, "r:bz2") as tar:
            tar.extractall(path=raw_dir_path, filter="fully_trusted")
            logging.info(f"Step 05 - Extracted files to: {raw_dir_path}")

    logging.info(f"Step 05 - Raw data directory path: {raw_dir_path}")
    logging.info("Step 05 - Completed")

    # Step 6: Parse xyz files
    logging.info("Step 06 - Parse xyz files")

    xyz_file_paths = list(raw_dir_path.rglob("*.xyz"))
    xyz_file_paths.sort()
    for xyz_file_path in xyz_file_paths:
        try:
            # Step 6a: Create data dictionary
            with open(xyz_file_path, "r") as xyz_file:
                xyz_str = xyz_file.read()
            data_dict = data_dict_from_xyz_str(xyz_str)
            # Step 6b: Remove center of gravity from coordinates
            data_dict["x"] = data_dict["x"] - torch.mean(data_dict["x"], dim=0, keepdim=True)
            if data_dict["x_ctx"] is not None:
                data_dict["x_ctx"] = data_dict["x_ctx"] - torch.mean(data_dict["x_ctx"], dim=0, keepdim=True)
            logging.info(f"Step 06 - Successfully processed '{xyz_file_path}'")
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logging.error(f"Step 06 - Failed to process file '{xyz_file_path}': {e}")
            continue

        break

    logging.info("Step 06 - Completed")


if __name__ == "__main__":
    main()
