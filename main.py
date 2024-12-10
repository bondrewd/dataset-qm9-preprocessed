from src.dataset import QM9Dataset


def main():
    dataset = QM9Dataset()
    print("QM9 Dataset")
    print(f"Length: {len(dataset)}")
    print(f"Location: {dataset.dataset_data_path.resolve()}")


if __name__ == "__main__":
    main()
