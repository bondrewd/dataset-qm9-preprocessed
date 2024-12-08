import itertools
from typing import Optional

import torch
from torch import Tensor


def onehot_from_element(element: str) -> Tensor:
    match element:
        case "H":
            return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        case "C":
            return torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        case "N":
            return torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        case "O":
            return torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        case "F":
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        case _:
            raise ValueError(f"Unknown element {element}")


def element_from_onehot(onehot: Tensor) -> str:
    if onehot.shape != (5,):
        raise Exception("Invalid onehot shape")
    if torch.sum(onehot) != 1.0 or torch.sum(onehot > 0.0) != 1:
        raise Exception("Invalid onehot format")
    if onehot[0] == 1.0:
        return "H"
    if onehot[1] == 1.0:
        return "C"
    if onehot[2] == 1.0:
        return "N"
    if onehot[3] == 1.0:
        return "O"
    if onehot[4] == 1.0:
        return "F"


def data_dict_from_xyz_str(xyz_str: str) -> dict[str, Optional[Tensor]]:
    # Read xyz file
    lines = xyz_str.splitlines()

    # Parse number of atoms
    num_nodes = int(lines[0])

    # Parse elements and coordinates
    elements = []
    coordinates = []
    for line in lines[2:num_nodes + 2]:
        # Tokenize line
        tokens = line.strip().split()
        # Parse element
        element = tokens[0].strip()
        element = onehot_from_element(element)
        # Parse coordinates
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3])
        coordinate = torch.tensor([[x, y, z]])
        # Save element and coordinates
        elements.append(element)
        coordinates.append(coordinate)

    # Generate all possible pairs of nodes
    edges = list(itertools.combinations(range(num_nodes), 2))
    # Separate the pairs into source and destination lists
    src, dst = zip(*edges)
    # Create a tensor of shape (2, x) where x is the number of edges
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.int64)

    data_dict = {
        "h": torch.vstack(elements),
        "x": torch.vstack(coordinates),
        "e": torch.tensor(edge_index),
        "a": None,
        "g": None,
        "h_ctx": None,
        "x_ctx": None,
        "e_ctx": None,
        "a_ctx": None,
        "g_ctx": None,
    }

    return data_dict


def xyz_str_from_data_dict(data_dict: dict[str, Optional[Tensor]]) -> str:
    xyz_str = ""

    h = data_dict["h"]
    x = data_dict["x"]

    assert h.shape[0] == x.shape[0], "Number of nodes does not match number of coordinates"

    xyz_str += f"{h.shape[0]}\n\n"

    for onehot, coordinate in zip(h, x):
        element = element_from_onehot(onehot)
        xyz_str += f"{element} {coordinate[0]:8.3f} {coordinate[1]:8.3f} {coordinate[2]:8.3f}\n"

    return xyz_str
