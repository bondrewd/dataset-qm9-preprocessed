from dataset_qm9_preprocessed.dataset import QM9Dataset


def test_dataset_len():
    dataset = QM9Dataset()
    assert len(dataset) == 133885, f"Expected dataset len to be 133885, got {len(dataset)}"


def test_dataset_getitem():
    dataset = QM9Dataset()

    data_dict = dataset[0]
    assert data_dict is not None, "Expected data dictionary, got None"

    h = data_dict["h"]
    x = data_dict["x"]
    e = data_dict["e"]
    a = data_dict["a"]
    g = data_dict["g"]
    h_ctx = data_dict["h_ctx"]
    x_ctx = data_dict["x_ctx"]
    e_ctx = data_dict["e_ctx"]
    a_ctx = data_dict["a_ctx"]
    g_ctx = data_dict["g_ctx"]

    assert h is not None, "Expected h, got None"
    assert x is not None, "Expected x, got None"
    assert e is not None, "Expected e, got None"
    assert a is None, f"Expected None, got {a}"
    assert g is None, f"Expected None, got {g}"
    assert h_ctx is None, f"Expected None, got {h_ctx}"
    assert x_ctx is None, f"Expected None, got {x_ctx}"
    assert e_ctx is None, f"Expected None, got {e_ctx}"
    assert a_ctx is None, f"Expected None, got {a_ctx}"
    assert g_ctx is None, f"Expected None, got {g_ctx}"

    assert h.shape[0] == x.shape[0], f"Expected h and x to be have the same first dimension, got h={h.shape[0]} and x={x.shape[0]}"
    assert x.shape[1] == 3, f"Expected x's second dimension to be 3, got {x.shape[1]}"
