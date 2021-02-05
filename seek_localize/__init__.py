"""Interface with iEEG-BIDS data and merge anatomical and electrophysiological data."""

import os

__name__ = "seek_localize"
__version__ = "0.1.0"

from .label import label_elecs_anat, convert_elecs_coords
from .bids import read_dig_bids
from .electrodes import Sensors
from .utils import read_fieldtrip_elecs

fs_lut_fpath = os.path.join(__name__, "templates/FreeSurferColorLUT.txt")
