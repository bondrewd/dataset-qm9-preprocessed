import argparse
import pathlib
import tarfile
import tomllib

import requests


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Load a configuration file in TOML format.")
    parser.add_argument("cfg", type=str, help="Path to the configuration file in TOML format.")
    args = parser.parse_args()

    # Read configuration file
    cfg_path = pathlib.Path(args.cfg)
    if not cfg_path.exists() or not cfg_path.is_file():
        raise FileNotFoundError(f"The configuration file '{cfg_path}' does not exist or is not a valid file.")

    with cfg_path.open("rb") as toml_file:
        cfg = tomllib.load(toml_file)

    print("Configuration loaded successfully:")
    print(cfg)

    # Create temporary directory
    tmp_dir_path = pathlib.Path("../tmp")
    tmp_dir_path.mkdir(parents=True, exist_ok=True)

    # Download raw data
    bz2_path = tmp_dir_path / "data.tar.bz2"
    if not bz2_path.exists():  # delete
        response = requests.get(cfg["url"])
        if response.status_code == 200:
            with open(bz2_path, "wb") as file:
                file.write(response.content)
            print(f"File downloaded: {bz2_path}")
        else:
            print(f"Failed to download file: {response.status_code}")

    # Decompress raw data
    raw_dir_path = tmp_dir_path / "raw"
    if not raw_dir_path.exists():  # delete
        raw_dir_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bz2_path, "r:bz2") as tar:
            tar.extractall(path=raw_dir_path, filter="fully_trusted")
            print(f"Extracted files to: {raw_dir_path}")


if __name__ == "__main__":
    main()
