import pytest
import torch


@pytest.fixture
def parameter_fixture():
    element_h = "H"
    element_c = "C"
    element_n = "N"
    element_o = "O"
    element_f = "F"
    onehot_h = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    onehot_c = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    onehot_n = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    onehot_o = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    onehot_f = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    onehot_shape = (1, 5)
    return {
        "element_h": element_h,
        "element_c": element_c,
        "element_n": element_n,
        "element_o": element_o,
        "element_f": element_f,
        "onehot_h": onehot_h,
        "onehot_c": onehot_c,
        "onehot_n": onehot_n,
        "onehot_o": onehot_o,
        "onehot_f": onehot_f,
        "onehot_shape": onehot_shape,
    }
