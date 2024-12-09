from fixtures import *
from src.utils import onehot_from_element, element_from_onehot, data_dict_from_xyz_str, xyz_str_from_data_dict


def test_onehot_from_element_execution(parameter_fixture):
    element_h = parameter_fixture["element_h"]
    try:
        onehot_from_element(element_h)
    except ValueError as e:
        pytest.fail(e)


def test_onehot_from_element_output_shape(parameter_fixture):
    element_h = parameter_fixture["element_h"]
    onehot_shape = parameter_fixture["onehot_shape"]
    onehot = onehot_from_element(element_h)
    assert onehot.shape == onehot_shape, "Output shape is incorrect"


def test_onehot_from_element_value(parameter_fixture):
    element_h = parameter_fixture["element_h"]
    element_c = parameter_fixture["element_c"]
    element_n = parameter_fixture["element_n"]
    element_o = parameter_fixture["element_o"]
    element_f = parameter_fixture["element_f"]
    onehot_h = parameter_fixture["onehot_h"]
    onehot_c = parameter_fixture["onehot_c"]
    onehot_n = parameter_fixture["onehot_n"]
    onehot_o = parameter_fixture["onehot_o"]
    onehot_f = parameter_fixture["onehot_f"]

    onehot = onehot_from_element(element_h)
    assert torch.equal(onehot, onehot_h), "One hot tensor is incorrect"

    onehot = onehot_from_element(element_c)
    assert torch.equal(onehot, onehot_c), "One hot tensor is incorrect"

    onehot = onehot_from_element(element_n)
    assert torch.equal(onehot, onehot_n), "One hot tensor is incorrect"

    onehot = onehot_from_element(element_o)
    assert torch.equal(onehot, onehot_o), "One hot tensor is incorrect"

    onehot = onehot_from_element(element_f)
    assert torch.equal(onehot, onehot_f), "One hot tensor is incorrect"


def test_onehot_from_element_raises():
    element = "A"
    with pytest.raises(ValueError, match=f"Unknown element {element}"):
        onehot_from_element(element)


def test_element_from_onehot_execution(parameter_fixture):
    onehot_h = parameter_fixture["onehot_h"]

    try:
        element_from_onehot(onehot_h)
    except ValueError as e:
        pytest.fail(e)


def test_element_from_onehot_value(parameter_fixture):
    element_h = parameter_fixture["element_h"]
    element_c = parameter_fixture["element_c"]
    element_n = parameter_fixture["element_n"]
    element_o = parameter_fixture["element_o"]
    element_f = parameter_fixture["element_f"]
    onehot_h = parameter_fixture["onehot_h"]
    onehot_c = parameter_fixture["onehot_c"]
    onehot_n = parameter_fixture["onehot_n"]
    onehot_o = parameter_fixture["onehot_o"]
    onehot_f = parameter_fixture["onehot_f"]

    element = element_from_onehot(onehot_h)
    assert element == element_h, "Element is incorrect"

    element = element_from_onehot(onehot_c)
    assert element == element_c, "Element is incorrect"

    element = element_from_onehot(onehot_n)
    assert element == element_n, "Element is incorrect"

    element = element_from_onehot(onehot_o)
    assert element == element_o, "Element is incorrect"

    element = element_from_onehot(onehot_f)
    assert element == element_f, "Element is incorrect"


def test_element_from_onehot_raises():
    with pytest.raises(ValueError, match="Invalid onehot shape"):
        element_from_onehot(torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="Invalid onehot format"):
        element_from_onehot(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="Invalid onehot format"):
        element_from_onehot(torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="Invalid onehot format"):
        element_from_onehot(torch.tensor([0.5, 0.0, 0.5, 0.0, 0.0]))


def test_data_dict_from_xyz_str_execution(xyz_fixture):
    xyz_str = xyz_fixture["xyz_str"]

    try:
        data_dict_from_xyz_str(xyz_str)
    except ValueError as e:
        pytest.fail(e)


def test_data_dict_from_xyz_str_output_shape(xyz_fixture):
    xyz_str = xyz_fixture["xyz_str"]
    xyz_data_dict = xyz_fixture["xyz_data_dict"]

    data_dict = data_dict_from_xyz_str(xyz_str)

    assert data_dict["h"].shape == xyz_data_dict["h"].shape, "Output shape is incorrect"
    assert data_dict["x"].shape == xyz_data_dict["x"].shape, "Output shape is incorrect"
    assert data_dict["e"].shape == xyz_data_dict["e"].shape, "Output shape is incorrect"


def test_data_dict_from_xyz_str_output_value(xyz_fixture):
    xyz_str = xyz_fixture["xyz_str"]
    xyz_data_dict = xyz_fixture["xyz_data_dict"]

    data_dict = data_dict_from_xyz_str(xyz_str)

    assert torch.equal(data_dict["h"], xyz_data_dict["h"]), "Output is incorrect"
    assert torch.equal(data_dict["x"], xyz_data_dict["x"]), "Output is incorrect"
    assert torch.equal(data_dict["e"], xyz_data_dict["e"]), "Output is incorrect"
    assert data_dict["a"] == xyz_data_dict["a"], "Output is incorrect"
    assert data_dict["g"] == xyz_data_dict["g"], "Output is incorrect"
    assert data_dict["h_ctx"] == xyz_data_dict["h_ctx"], "Output is incorrect"
    assert data_dict["x_ctx"] == xyz_data_dict["x_ctx"], "Output is incorrect"
    assert data_dict["e_ctx"] == xyz_data_dict["e_ctx"], "Output is incorrect"
    assert data_dict["a_ctx"] == xyz_data_dict["a_ctx"], "Output is incorrect"
    assert data_dict["g_ctx"] == xyz_data_dict["g_ctx"], "Output is incorrect"


def test_xyz_str_from_data_dict_execution(xyz_fixture):
    xyz_data_dict = xyz_fixture["xyz_data_dict"]

    try:
        xyz_str_from_data_dict(xyz_data_dict)
    except ValueError as e:
        pytest.fail(e)


def test_xyz_str_from_data_dict_output_value(xyz_fixture):
    xyz_data_dict = xyz_fixture["xyz_data_dict"]

    xyz_str = xyz_str_from_data_dict(xyz_data_dict)
    xyz_lines = xyz_str.splitlines()

    assert xyz_lines[0] == "18"
    assert xyz_lines[1] == ""
    assert xyz_lines[2] == "C    5.332    5.901   -2.448"
    assert xyz_lines[3] == "N    4.766    4.809   -3.227"
    assert xyz_lines[4] == "C    3.410    4.275   -3.077"
    assert xyz_lines[5] == "C    4.157    3.377   -1.281"
    assert xyz_lines[6] == "C    3.066    2.425   -1.606"
    assert xyz_lines[7] == "C    3.427    1.585   -2.612"
    assert xyz_lines[8] == "C    4.793    2.012   -3.185"
    assert xyz_lines[9] == "C    4.677    3.493   -2.709"
    assert xyz_lines[10] == "C    2.819    3.947   -1.680"
    assert xyz_lines[11] == "H    4.943    6.848   -2.833"
    assert xyz_lines[12] == "H    6.419    5.897   -2.575"
    assert xyz_lines[13] == "H    5.106    5.833   -1.375"
    assert xyz_lines[14] == "H    2.856    4.158   -4.003"
    assert xyz_lines[15] == "H    4.677    3.417   -0.333"
    assert xyz_lines[16] == "H    2.740    0.990   -3.208"
    assert xyz_lines[17] == "H    4.810    1.865   -4.270"
    assert xyz_lines[18] == "H    5.683    1.528   -2.765"
    assert xyz_lines[19] == "H    2.095    4.510   -1.101"


def test_inverse(xyz_fixture):
    xyz_str_original = xyz_str_from_data_dict(xyz_fixture["xyz_data_dict"])
    xyz_data_dict_original = data_dict_from_xyz_str(xyz_str_original)

    xyz_str_inverted = xyz_str_from_data_dict(data_dict_from_xyz_str(xyz_str_original))
    assert xyz_str_original == xyz_str_inverted, "Inverse failed"

    xyz_data_dict_inverted = data_dict_from_xyz_str(xyz_str_from_data_dict(xyz_data_dict_original))
    assert torch.equal(xyz_data_dict_original["h"], xyz_data_dict_inverted["h"]), "Inverse failed"
    assert torch.equal(xyz_data_dict_original["x"], xyz_data_dict_inverted["x"]), "Inverse failed"
    assert torch.equal(xyz_data_dict_original["e"], xyz_data_dict_inverted["e"]), "Inverse failed"
    assert xyz_data_dict_original["a"] == xyz_data_dict_inverted["a"], "Inversibility failed"
    assert xyz_data_dict_original["h_ctx"] == xyz_data_dict_inverted["h_ctx"], "Inverse failed"
    assert xyz_data_dict_original["x_ctx"] == xyz_data_dict_inverted["x_ctx"], "Inverse failed"
    assert xyz_data_dict_original["e_ctx"] == xyz_data_dict_inverted["e_ctx"], "Inverse failed"
    assert xyz_data_dict_original["a_ctx"] == xyz_data_dict_inverted["a_ctx"], "Inverse failed"
