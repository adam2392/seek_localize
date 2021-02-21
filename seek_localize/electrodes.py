from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import List, Union

import numpy as np


def _pl(x, non_pl=""):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, (int, np.generic)) else len(x)
    return non_pl if len_x == 1 else "s"


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

    def __repr__(self):
        """Taken from mne-python."""
        MAX_WIDTH = 68
        strs = ["<Sensors | %s non-empty values"]
        non_empty = 0
        for k, v in self.__dict__.items():
            if k == "ch_names":
                if v:
                    entr = shorten(", ".join(v), MAX_WIDTH, placeholder=" ...")
                else:
                    entr = "[]"  # always show
                    non_empty -= 1  # don't count as non-empty
            elif k in ["coord_system", "coord_unit"]:
                entr = v
            else:
                try:
                    this_len = len(v)
                except TypeError:
                    entr = "{}".format(v) if v is not None else ""
                else:
                    if this_len > 0:
                        entr = "%d item%s (%s)" % (
                            this_len,
                            _pl(this_len),
                            type(v).__name__,
                        )
                    else:
                        entr = ""
            if entr != "":
                non_empty += 1
                strs.append("%s: %s" % (k, entr))
        st = "\n ".join(sorted(strs))
        st += "\n>"
        st %= non_empty
        return st

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
