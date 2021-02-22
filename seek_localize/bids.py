import json
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from mne.utils import run_subprocess
from mne_bids import get_entities_from_fname, BIDSPath
from mne_bids.config import BIDS_COORDINATE_UNITS, COORD_FRAME_DESCRIPTIONS
from mne_bids.tsv_handler import _from_tsv
from mne_bids.utils import _write_json, _write_tsv


def _suffix_chop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s


def _match_dig_sidecars(bids_path):
    """Match the sidecar files that define iEEG montage.

    Returns the filepath to ``electrodes.tsv`` and
    ``coordsystem.json`` files.
    """
    electrodes_fnames = BIDSPath(
        subject=bids_path.subject,
        session=bids_path.session,
        space=bids_path.space,
        suffix="electrodes",
        extension=".tsv",
        root=bids_path.root,
    ).match()

    if len(electrodes_fnames) != 1:
        raise ValueError(
            f"Trying to load electrodes.tsv "
            f"file using {bids_path}, but there "
            f"are multiple files "
            f"electrodes.tsv. Please "
            f"uniquely specify it using additional "
            f"BIDS entities."
        )
    elec_fname = electrodes_fnames[0]
    coord_fname = elec_fname.copy().update(suffix="coordsystem", extension=".json")
    if not coord_fname.fpath.exists():
        raise ValueError(
            f"Corresponding coordsystem.json file does not exist "
            f"for {elec_fname}. Please check BIDS dataset."
        )
    return elec_fname, coord_fname


def bids_validate(bids_root):
    """Run BIDS validator."""
    shell = False
    bids_validator_exe = ["bids-validator", "--config.error=41", "--conwfig.error=41"]
    if platform.system() == "Windows":
        shell = True
        exe = os.getenv("VALIDATOR_EXECUTABLE", "n/a")
        if "VALIDATOR_EXECUTABLE" != "n/a":
            bids_validator_exe = ["node", exe]

    def _validate(bids_root):
        cmd = bids_validator_exe + [bids_root]
        run_subprocess(cmd, shell=shell)

    return _validate(bids_root)


def _coordsystem_json(
    unit, sensor_coord_system, intended_for, fname, overwrite=False, verbose=True
):
    """Create a coordsystem.json file and save it.

    Parameters
    ----------
    unit : str
        Units to be used in the coordsystem specification,
        as in BIDS_COORDINATE_UNITS.
    sensor_coord_system : str
        Name of the coordinate system for the sensor positions.
    fname : str
        Filename to save the coordsystem.json to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.
    """
    if unit not in BIDS_COORDINATE_UNITS:
        raise ValueError(
            f"Units of the sensor positions must be one "
            f"of {BIDS_COORDINATE_UNITS}. You passed in "
            f"{unit}."
        )

    if sensor_coord_system == "Image" and intended_for is None:
        raise RuntimeError(
            'If coordsystem is "Image", then img_fname '
            "must be passed in as the filename of the corresponding "
            "image the data is in. For example"
        )

    # get the coordinate frame description
    sensor_coord_system_descr = COORD_FRAME_DESCRIPTIONS.get(
        sensor_coord_system.lower(), "n/a"
    )
    if sensor_coord_system == "Other" and verbose:
        print(
            "Using the `Other` keyword for the CoordinateSystem field. "
            "Please specify the CoordinateSystemDescription field manually."
        )

    if sensor_coord_system == "Other" and sensor_coord_system_descr == "n/a":
        raise RuntimeError(
            'If coordsystem is "Other", then coordsystem_description '
            "must be passed in describing the coordinate system "
            "in Free-form text. May also include a link to a "
            "documentation page or paper describing the system in "
            "greater detail."
        )

    processing_description = (
        "SEEK-algorithm (thresholding, cylindrical clustering and post-processing), "
        "or manual labeling of contacts using FieldTrip Toolbox."
    )

    # create the coordinate json data structure based on 'datatype'
    fid_json = {
        # (Other, Pixels, ACPC)
        "IntendedFor": str(intended_for),
        "iEEGCoordinateSystem": sensor_coord_system,
        "iEEGCoordinateSystemDescription": sensor_coord_system_descr,
        "iEEGCoordinateUnits": unit,  # m (MNE), mm, cm , or pixels
        "iEEGCoordinateProcessingDescription": processing_description,
        "iEEGCoordinateProcessingReference": "See DOI: https://zenodo.org/record/3542307#.XoYF9tNKhZI "
        "and FieldTrip Toolbox: doi:10.1155/2011/156869",
    }

    # note that any coordsystem.json file shared within sessions
    # will be the same across all runs (currently). So
    # overwrite is set to True always
    # XXX: improve later when BIDS is updated
    # check that there already exists a coordsystem.json
    if Path(fname).exists() and not overwrite:
        with open(fname, "r", encoding="utf-8-sig") as fin:
            coordsystem_dict = json.load(fin)
        if fid_json != coordsystem_dict:
            raise RuntimeError(
                f"Trying to write coordsystem.json, but it already "
                f"exists at {fname} and the contents do not match. "
                f"You must differentiate this coordsystem.json file "
                f'from the existing one, or set "overwrite" to True.'
            )
    _write_json(fname, fid_json, overwrite=True, verbose=verbose)


def _electrodes_tsv(ch_names, ch_coords, fname, overwrite=False, verbose=True):
    """Create an electrodes.tsv file and save it.

    Parameters
    ----------
    ch_names : list | np.ndarray
        Name of the electrode contact point. Corresponds to
        ``name`` in iEEG-BIDS for electrodes.tsv file.
    ch_coords : list | np.ndarray
        List of sensor xyz positions. Corresponds to
        ``x``, ``y``, ``z`` in iEEG-BIDS for electrodes.tsv file.
    fname : str
        Filename to save the electrodes.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.
    """
    # create list of channel coordinates and names
    x, y, z, names = list(), list(), list(), list()
    for ch, coord in zip(ch_names, ch_coords):
        if any(np.isnan(_coord) for _coord in coord):
            x.append("n/a")
            y.append("n/a")
            z.append("n/a")
        else:
            x.append(coord[0])
            y.append(coord[1])
            z.append(coord[2])
        names.append(ch)

    # create OrderedDict to write to tsv file
    # XXX: size should be included in the future
    sizes = ["n/a"] * len(names)
    data = OrderedDict(
        [
            ("name", names),
            ("x", x),
            ("y", y),
            ("z", z),
            ("size", sizes),
        ]
    )

    # note that any coordsystem.json file shared within sessions
    # will be the same across all runs (currently). So
    # overwrite is set to True always
    # XXX: improve later when BIDS is updated
    # check that there already exists a coordsystem.json
    if Path(fname).exists() and not overwrite:
        electrodes_tsv = _from_tsv(fname)

        # cast values to str to make equality check work
        if any(
            [
                list(map(str, vals1)) != list(vals2)
                for vals1, vals2 in zip(data.values(), electrodes_tsv.values())
            ]
        ):
            raise RuntimeError(
                f"Trying to write electrodes.tsv, but it already "
                f"exists at {fname} and the contents do not match. "
                f"You must differentiate this electrodes.tsv file "
                f'from the existing one, or set "overwrite" to True.'
            )
    _write_tsv(fname, data, overwrite=True, verbose=verbose)


def write_dig_bids(
    fname: BIDSPath,
    root,
    ch_names: List,
    ch_coords: List,
    unit: str,
    coord_system: str,
    intended_for: Union[str, Path],
    sizes: List = None,
    groups: List = None,
    hemispheres: List = None,
    manufacturers: List = None,
    overwrite: bool = False,
    verbose: bool = True,
):
    """Write iEEG-BIDS coordinates and related files to disc.

    Parameters
    ----------
    fname : str
    root : str
    ch_names : list
    ch_coords : list
    unit : str
    coord_system : str
    intended_for : str
    sizes : list | None
    groups : list | None
    hemispheres : list | None
    manufacturers : list | None
    overwrite : bool
    verbose : bool
        Verbosity
    """
    # check
    _checklen = len(ch_names)
    if not all(len(_check) == _checklen for _check in [ch_names, ch_coords]):
        raise ValueError(
            "Number of channel names should match "
            "number of coordinates passed in. "
            f"{len(ch_names)} names and {len(ch_coords)} coords passed in."
        )

    for name, _check in zip(
        ["size", "group", "hemisphere", "manufacturer"],
        [sizes, groups, hemispheres, manufacturers],
    ):
        if _check is not None:
            if len(_check) != _checklen:
                raise ValueError(
                    f"Number of {name} ({len(_check)} should match "
                    f"number of channel names passed in "
                    f"({len(ch_names)} names)."
                )

    # check that filename adheres to BIDS naming convention
    entities = get_entities_from_fname(fname, on_error="raise")

    # get the 3 channels needed
    datatype = "ieeg"
    elecs_fname = BIDSPath(**entities, datatype=datatype, root=root)
    elecs_fname.update(suffix="electrodes", extension=".tsv")
    coordsys_fname = elecs_fname.copy().update(suffix="coordsystem", extension=".json")

    # make parent directories
    Path(elecs_fname).parent.mkdir(exist_ok=True, parents=True)

    # write the coordsystem json file
    _coordsystem_json(
        unit,
        coord_system,
        intended_for,
        fname=coordsys_fname,
        overwrite=overwrite,
        verbose=verbose,
    )

    # write basic electrodes tsv file
    _electrodes_tsv(
        ch_names=ch_names,
        ch_coords=ch_coords,
        fname=elecs_fname,
        overwrite=overwrite,
        verbose=verbose,
    )

    for name, _check in zip(
        ["size", "group", "hemisphere", "manufacturer"],
        [sizes, groups, hemispheres, manufacturers],
    ):
        if _check is not None:
            # write this additional data now to electrodes.tsv
            elec_df = pd.read_csv(elecs_fname, delimiter="\t", index=None)
            elec_df[name] = _check
            elec_df.to_csv(elecs_fname, index=None, sep="\t")

            # for groups, these need to match inside the channels data
            if name == "group":
                chs_fname = elecs_fname.copy().update(suffix="channels")
                chs_df = pd.read_csv(chs_fname, delimiter="\t", index=None)
                chs_df[name] = _check
                chs_df.to_csv(chs_fname, index=None, sep="\t")
