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
        raise ValueError("Invalid onehot shape")
    if torch.sum(onehot) != 1.0 or torch.sum(onehot > 0.0) != 1:
        raise ValueError("Invalid onehot format")
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


def data_dict_from_xyz_str(xyz_str: str) -> dict[str, Optional[Tensor] | list[int]]:
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
        x = float(tokens[1].replace('*^', 'e'))
        y = float(tokens[2].replace('*^', 'e'))
        z = float(tokens[3].replace('*^', 'e'))
        coordinate = torch.tensor([[x, y, z]])
        # Save element and coordinates
        elements.append(element)
        coordinates.append(coordinate)

    # Calculate edges
    if num_nodes > 1:
        # Generate all possible pairs of nodes
        edges = list(itertools.combinations(range(num_nodes), 2))
        # Separate the pairs into source and destination lists
        src, dst = zip(*edges)
        # Create a tensor of shape (2, x) where x is the number of edges
        edge_index = torch.tensor([src + dst, dst + src])
    else:
        edge_index = None

    # Calculate segments
    segments = torch.tensor([num_nodes])

    data_dict = {
        "h": torch.vstack(elements),
        "x": torch.vstack(coordinates),
        "e": edge_index,
        "a": None,
        "g": None,
        "h_ctx": None,
        "x_ctx": None,
        "e_ctx": None,
        "a_ctx": None,
        "g_ctx": None,
        "segments": segments,
    }

    return data_dict


def xyz_str_from_data_dict(data_dict: dict[str, Optional[Tensor] | list[int]]) -> str:
    xyz_str = ""

    n = data_dict["segments"].item()
    h = data_dict["h"]
    x = data_dict["x"]

    assert h.shape[0] == x.shape[0], "Number of nodes does not match number of coordinates"

    xyz_str += f"{n}\n\n"

    for onehot, coordinate in zip(h, x):
        element = element_from_onehot(onehot)
        xyz_str += f"{element} {coordinate[0]:8.3f} {coordinate[1]:8.3f} {coordinate[2]:8.3f}\n"

    return xyz_str


def collate_data_dicts(data_dicts: list[dict[str, Optional[Tensor] | list[int]]]) -> dict[str, Optional[Tensor] | list[int]]:
    # Concatenate segments
    segments = torch.cat([data_dict["segments"] for data_dict in data_dicts])

    # Concatenate nodes features
    h = torch.cat([data_dict["h"] for data_dict in data_dicts], dim=0)

    # Concatenate positions
    x = torch.cat([data_dict["x"] for data_dict in data_dicts], dim=0)

    # Concatenate edges
    offsets = [0] + segments.tolist()[:-1]
    e = torch.cat([data_dict["e"] + offset for data_dict, offset in zip(data_dicts, offsets)], dim=1)

    # Concatenate edge features
    if data_dicts[0]["a"] is not None:
        a = torch.cat([data_dict["a"] for data_dict in data_dicts], dim=0)
    else:
        a = None

    # Concatenate graph features
    if data_dicts[0]["g"] is not None:
        g = torch.cat([data_dict["g"] for data_dict in data_dicts], dim=0)
    else:
        g = None

    # Concatenate context node features
    if data_dicts[0]["h_ctx"] is not None:
        h_ctx = torch.cat([data_dict["h_ctx"] for data_dict in data_dicts], dim=0)
    else:
        h_ctx = None

    # Concatenate context node features
    if data_dicts[0]["x_ctx"] is not None:
        x_ctx = torch.cat([data_dict["x_ctx"] for data_dict in data_dicts], dim=0)
    else:
        x_ctx = None

    # Concatenate edges
    if data_dicts[0]["e_ctx"] is not None:
        ctx_segments = [data_dict["h_ctx"].shape[0] for data_dict in data_dicts]
        ctx_offsets = [0] + ctx_segments[:-1]
        global_offset = sum(segments)
        e_ctx = torch.cat([torch.cat([
            data_dict["e_ctx"][:1] + offset,
            data_dict["e_ctx"][1:] - data_dict["h"].shape[0] + global_offset + ctx_offset,
        ], dim=0) for data_dict, offset, ctx_offset in zip(data_dicts, offsets, ctx_offsets)], dim=1)
    else:
        e_ctx = None

    # Concatenate context node features
    if data_dicts[0]["a_ctx"] is not None:
        a_ctx = torch.cat([data_dict["a_ctx"] for data_dict in data_dicts], dim=0)
    else:
        a_ctx = None

    # Concatenate context node features
    if data_dicts[0]["g_ctx"] is not None:
        g_ctx = torch.cat([data_dict["g_ctx"] for data_dict in data_dicts], dim=0)
    else:
        g_ctx = None

    return {
        "h": h,
        "x": x,
        "e": e,
        "a": a,
        "g": g,
        "h_ctx": h_ctx,
        "x_ctx": x_ctx,
        "e_ctx": e_ctx,
        "a_ctx": a_ctx,
        "g_ctx": g_ctx,
        "segments": segments,
    }
