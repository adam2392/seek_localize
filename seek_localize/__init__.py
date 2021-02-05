"""Interface with iEEG-BIDS data and merge anatomical and electrophysiological data."""

import os

__name__ = "seek_localize"
__version__ = "0.1.0"

from .label import label_elecs_anat
from .bids import bids_validate, write_coordsystem_json, write_electrodes_tsv
from .utils import read_fieldtrip_elecs

fs_lut_fpath = os.path.join(__name__, "templates/FreeSurferColorLUT.txt")
