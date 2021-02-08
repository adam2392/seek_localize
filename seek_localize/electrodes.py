from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np


@dataclass()
class Sensors:
    """Sensor data class."""

    ch_names: List
    x: List
    y: List
    z: List
    coord_system: str
    coord_unit: str
    elecs_fname: str
    coordsystem_fname: str
    intended_for: Union[str, Path, None] = None

    def get_coords(self):
        """Get coordinates as a N x 3 array."""
        return np.vstack((self.x, self.y, self.z)).astype(float).T

    def set_coords(self, coords):
        """Set coordinates from an array, or dictionary."""
        if isinstance(coords, np.ndarray):
            self.x = coords[:, 0]
            self.y = coords[:, 1]
            self.z = coords[:, 2]
        else:
            raise NotImplementedError("not done yet...")

    def as_dict(self):
        """Return coordinates as a dictionary of name: 3D coord."""
        data = OrderedDict(
            [
                ("name", self.ch_names),
                ("x", self.x),
                ("y", self.y),
                ("z", self.z),
            ]
        )
        return data
