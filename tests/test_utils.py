from fixtures import *
from src.utils import onehot_from_element, element_from_onehot, xyz_to_data_dict


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


def test_xyz_to_data_dict_execution(tmp_path, xyz_fixture):
    xyz_file_content = xyz_fixture["xyz_file_content"]

    tmp_xyz_path = tmp_path / "test_file.xyz"
    tmp_xyz_path.write_text(xyz_file_content)
    tmp_xyz_path_str = str(tmp_xyz_path)

    try:
        xyz_to_data_dict(tmp_xyz_path_str)
    except ValueError as e:
        pytest.fail(e)


def test_xyz_to_data_dict_output_shape(tmp_path, xyz_fixture):
    xyz_file_content = xyz_fixture["xyz_file_content"]
    xyz_data_dict = xyz_fixture["xyz_data_dict"]

    tmp_xyz_path = tmp_path / "test_file.xyz"
    tmp_xyz_path.write_text(xyz_file_content)
    tmp_xyz_path_str = str(tmp_xyz_path)

    data_dict = xyz_to_data_dict(tmp_xyz_path_str)

    assert data_dict["h"].shape == xyz_data_dict["h"].shape, "Output shape is incorrect"
    assert data_dict["x"].shape == xyz_data_dict["x"].shape, "Output shape is incorrect"
    assert data_dict["e"].shape == xyz_data_dict["e"].shape, "Output shape is incorrect"


def test_xyz_to_data_dict_output_value(tmp_path, xyz_fixture):
    xyz_file_content = xyz_fixture["xyz_file_content"]
    xyz_data_dict = xyz_fixture["xyz_data_dict"]

    tmp_xyz_path = tmp_path / "test_file.xyz"
    tmp_xyz_path.write_text(xyz_file_content)
    tmp_xyz_path_str = str(tmp_xyz_path)

    data_dict = xyz_to_data_dict(tmp_xyz_path_str)

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
