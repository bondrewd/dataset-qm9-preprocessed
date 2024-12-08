import argparse
import logging
import pathlib
import shutil
import tarfile
import tempfile
import tomllib

import requests
import torch

from src.utils import data_dict_from_xyz_str


def main():
    # Setup logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # ---
    # Step 1: Parse arguments
    # ---
    logging.info("Step 01 - Parse arguments")

    parser = argparse.ArgumentParser(description="Load a configuration file in TOML format.")
    parser.add_argument("cfg", type=str, help="Path to the configuration file in TOML format")
    parser.add_argument("--output", "-o", type=str, help="Output database name", default="dataset-qm9")
    parser.add_argument("--tmp", type=str, help="Temporary directory for downloading and processing data")
    parser.add_argument("--tmp-keep", action="store_true", help="Remove temporary directory after processing data")

    args = parser.parse_args()

    logging.info("Step 01 - Completed")

    # ---
    # Step 2: Read configuration file
    # ---
    logging.info("Step 02 - Read configuration file")

    cfg_path = pathlib.Path(args.cfg)
    if not cfg_path.exists() or not cfg_path.is_file():
        logging.error(f"Step 02 - The configuration file '{cfg_path}' does not exist or is not a valid file.")

    with cfg_path.open("rb") as toml_file:
        cfg = tomllib.load(toml_file)

    logging.info("Step 02 - Completed")

    # ---
    # Step 3: Create temporary directory
    # ---
    logging.info("Step 03 - Create temporary directory")

    if args.tmp:
        tmp_dir_path = pathlib.Path(args.tmp)
        tmp_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_dir_path = pathlib.Path(tmp_dir.name)

    logging.info("Step 03 - Completed")

    # ---
    # Step 4: Download raw data
    # ---
    logging.info("Step 04 - Download raw data")

    bz2_path = tmp_dir_path / "data.tar.bz2"
    raw_data_url = cfg["url"]
    if not bz2_path.exists():  # delete
        response = requests.get(raw_data_url)
        if response.status_code == 200:
            with open(bz2_path, "wb") as file:
                file.write(response.content)
            logging.info(f"Step 04 - File downloaded: {bz2_path}")
        else:
            logging.error(f"Step 04 - Failed to download file: {response.status_code}")
            return

    logging.info(f"Step 04 - Raw data url: {raw_data_url}")
    logging.info(f"Step 04 - Raw data path: {bz2_path}")
    logging.info("Step 04 - Completed")

    # ---
    # Step 5: Decompress raw data
    # ---
    logging.info("Step 05 - Decompress raw data")

    raw_dir_path = tmp_dir_path / "raw"
    if not raw_dir_path.exists():  # delete
        raw_dir_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bz2_path, "r:bz2") as tar:
            tar.extractall(path=raw_dir_path, filter="fully_trusted")
            logging.info(f"Step 05 - Extracted files to: {raw_dir_path}")

    logging.info(f"Step 05 - Raw data directory path: {raw_dir_path}")
    logging.info("Step 05 - Completed")

    # ---
    # Step 6: Parse xyz files
    # ---
    logging.info("Step 07 - List and sort xyz files")

    xyz_file_paths = list(raw_dir_path.rglob("*.xyz"))
    xyz_file_paths.sort()

    logging.info("Step 06 - Completed")

    # ---
    # Step 7: Parse xyz files
    # ---
    logging.info("Step 07 - Parse xyz files")

    data_dicts = {}
    for xyz_file_path in xyz_file_paths:
        try:
            # Step 7a: Create data dictionary
            with open(xyz_file_path, "r") as xyz_file:
                xyz_str = xyz_file.read()
            data_dict = data_dict_from_xyz_str(xyz_str)
            # Step 7b: Remove center of gravity from coordinates
            data_dict["x"] = data_dict["x"] - torch.mean(data_dict["x"], dim=0, keepdim=True)
            if data_dict["x_ctx"] is not None:
                data_dict["x_ctx"] = data_dict["x_ctx"] - torch.mean(data_dict["x_ctx"], dim=0, keepdim=True)
            # Step 7c: Save data dict
            data_dicts[xyz_file_path.stem] = data_dict
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logging.error(f"Step 07 - Failed to process file '{xyz_file_path}': {e}")
            continue

    logging.info("Step 07 - Completed")

    # ---
    # Step 8: Save processed data
    # ---
    logging.info("Step 08 - Save processed data")

    out_path = pathlib.Path(args.output).with_suffix(".pth")
    torch.save(data_dicts, out_path)

    logging.info("Step 08 - Completed")

    if not args.tmp_keep:
        # ---
        # Step 9: Cleanup tmp directory
        # ---
        logging.info("Step 09 - Cleanup tmp directory")

        if args.tmp:
            shutil.rmtree(tmp_dir_path)
        else:
            tmp_dir.cleanup()

        logging.info("Step 09 - Completed")

        logging.info("Done!")


if __name__ == "__main__":
    main()
