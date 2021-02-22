"""Interface with iEEG-BIDS data and merge anatomical and electrophysiological data."""

import os

__name__ = "seek_localize"
__version__ = "0.2.0"

import seek_localize
from .label import label_elecs_anat
from .coordsystem import convert_coord_space, convert_coord_units
from .io import read_dig_bids
from .electrodes import Sensors
from .utils import read_fieldtrip_elecs
from .bids import write_dig_bids

fs_lut_fpath = os.path.join(
    os.path.dirname(seek_localize.__file__), "templates/FreeSurferColorLUT.txt"
)
