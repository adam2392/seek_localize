"""Interface with iEEG-BIDS data and merge anatomical and electrophysiological data."""

import os

__name__ = "seek_localize"
__version__ = "0.1.0"

import seek_localize
from .label import label_elecs_anat, convert_elecs_coords
from .io import read_dig_bids
from .electrodes import Sensors
from .utils import read_fieldtrip_elecs

fs_lut_fpath = os.path.join(
    os.path.dirname(seek_localize.__file__), "templates/FreeSurferColorLUT.txt"
)
