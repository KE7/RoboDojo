"""Isaac Lab asset data as RoboDojo reads it: torch tensors, and poses with wxyz quaternions.

Isaac Lab 3.0 returns asset data as ``ProxyArray`` (``.torch`` is the tensor) and orders quaternions xyzw; Isaac Lab
2.x returned tensors with wxyz quaternions. RoboDojo's poses are ``[x, y, z, qw, qx, qy, qz]`` throughout.
"""

import torch


def tensor(data, name: str) -> torch.Tensor:
    """The field ``name`` of an asset's data as a torch tensor."""
    value = getattr(data, name)
    return value.torch if hasattr(value, "torch") else value


def pose_wxyz(data, name: str) -> torch.Tensor:
    """A pose field (``[..., 7]``: position, quaternion) with its quaternion in wxyz order."""
    value = getattr(data, name)
    if not hasattr(value, "torch"):                 # Isaac Lab 2.x: already wxyz
        return value
    pose = value.torch                              # Isaac Lab 3.x: xyzw
    return torch.cat([pose[..., :3], pose[..., 6:7], pose[..., 3:6]], dim=-1)


def isaac_lab_quat(wxyz):
    """A wxyz quaternion in the order Isaac Lab's configurations take: xyzw on Isaac Lab 3.x, wxyz on 2.x."""
    import importlib.util

    w, x, y, z = (float(v) for v in wxyz)
    return (x, y, z, w) if importlib.util.find_spec("isaaclab_physx") is not None else (w, x, y, z)
