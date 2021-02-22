import json

from mne_bids import get_entities_from_fname, BIDSPath
from mne_bids.tsv_handler import _from_tsv

from seek_localize.bids import _suffix_chop
from seek_localize.electrodes import Sensors


def _read_lut_file(lut_fname):
    """Read the FreeSurfer Lookup Table file.

    Creates a dictionary of labels per row index.
    """
    fid = open(lut_fname)
    LUT = fid.readlines()
    fid.close()

    # Make dictionary of labels
    LUT = [row.split() for row in LUT]
    lab = {}
    for row in LUT:
        if (
            len(row) > 1 and row[0][0] != "#" and row[0][0] != "\\"
        ):  # Get rid of the comments
            lname = row[1]
            lab[int(row[0])] = lname

    return lab


def _read_elecs_tsv(elecs_fname, as_dict=False):
    """Read BIDS electrodes.tsv file in.

    Only reads in ch names and coordinates.
    """
    # read in elecs_fname
    elecs_tsv = _from_tsv(elecs_fname)

    x, y, z = [], [], []
    ch_names = []
    x_coord = elecs_tsv["x"]
    y_coord = elecs_tsv["y"]
    z_coord = elecs_tsv["z"]
    for idx, ch_name in enumerate(elecs_tsv["name"]):
        ch_names.append(ch_name)
        x.append(x_coord[idx])
        y.append(y_coord[idx])
        z.append(z_coord[idx])

    if as_dict:
        ch_coords = {
            ch_name: (_x, _y, _z) for ch_name, _x, _y, _z in zip(ch_names, x, y, z)
        }
        return ch_coords

    return ch_names, x, y, z


def _read_coords_json(coordsystem_fname):
    """Read BIDS coordsystem.json file in.

    Only reads in the ``iEEGCoordinateSystem``.
    TODO: add better checking for coordinate system and frames
    from MNE <-> BIDS
    """
    with open(coordsystem_fname, "r") as fin:
        coordsystem_json = json.load(fin)
    coord_system = coordsystem_json["iEEGCoordinateSystem"]
    unit = coordsystem_json["iEEGCoordinateUnits"]
    # if coord_frame not in ACCEPTED_MNE_COORD_FRAMES:
    #     raise ValueError(f'Coord frame {coord_frame} '
    #                      f'from mne-python is not supported yet... '
    #                      f'Please make sure coordinate frame is one of '
    #                      f'{ACCEPTED_MNE_COORD_FRAMES}.')

    # convvert coordinate frame to coordinate system string
    # coord_system = MNE_TO_BIDS_FRAMES.get(coord_frame)
    return coord_system, unit


def read_dig_bids(fname, root, intended_for: str = None):
    """Read electrode coordinates from BIDS files.

    TODO: improve to error check coordinatesystem.

    Parameters
    ----------
    fname : str | pathlib.Path
        File path to the electrodes.tsv file, or
        file path to the coordsystem.json file. Each one will be inferred
        based on the BIDS entities inside the filename.
    intended_for : str | None
        Optional parameter to tell function path of the Nifti image
        to interpret sensor coordinates for.

    Returns
    -------
    sensors : seek_localize.Sensors
        A data class containing the electrode sensors.
    """
    # check that filename adheres to BIDS naming convention
    entities = get_entities_from_fname(fname, on_error="raise")

    # get the 3 channels needed
    datatype = "ieeg"
    elecs_fname = BIDSPath(**entities, datatype=datatype, root=root)
    elecs_fname.update(suffix="electrodes", extension=".tsv")
    coordsystem_fname = elecs_fname.copy().update(
        suffix="coordsystem", extension=".json"
    )

    # read in elecs_fname
    ch_names, x, y, z = _read_elecs_tsv(elecs_fname)

    # read in the coordinate system json
    # coord_system, unit = _read_coords_json(coordsystem_fname)
    with open(coordsystem_fname, "r") as fin:
        coordsystem_json = json.load(fin)
    coord_system = coordsystem_json["iEEGCoordinateSystem"]
    unit = coordsystem_json["iEEGCoordinateUnits"]

    # if units are voxels and coord_system is other, then
    # assume coordinate system 'mri'
    if coord_system == "other":
        print(
            "SETTING COORDINATE SYSTEM AS MRI by default if "
            'coordinatesystem is "other".'
        )
        coord_system = "mri"

    # get the BIDS root
    entities = get_entities_from_fname(elecs_fname)
    elecs_bids_path = BIDSPath(**entities, datatype="ieeg", extension=".tsv").fpath
    root = _suffix_chop(str(elecs_fname), str(elecs_bids_path))

    # try to get the BIDS Path to the image file that
    # the coordinates are intended to be interpreted for
    intended_for_fname = coordsystem_json.get("IntendedFor")
    if intended_for_fname:
        entities = get_entities_from_fname(intended_for_fname)
        intended_img_path = BIDSPath(**entities, root=root)
    elif intended_for is not None:
        intended_img_path = intended_for
    else:
        raise RuntimeError(
            f"IntendedFor inside {coordsystem_fname} "
            f"is not available, and ``intended_for`` "
            f"kwarg is not set. Please set "
            f"``intended_for``."
        )

    # get the coordinates and set
    # montage_coords = montage.get_positions()
    # ch_pos = montage_coords.get('ch_pos')
    # for ch_name, coord in ch_pos.items():
    #     ch_names.append(ch_name)
    #     x.append(coord[0])
    #     y.append(coord[1])
    #     z.append(coord[2])

    # get the coordinate frame in mne
    # XXX: Possible coordframes subject to change in mne
    # mri=FIFF.FIFFV_COORD_MRI,
    # mri_voxel=FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
    # head=FIFF.FIFFV_COORD_HEAD,
    # mni_tal=FIFF.FIFFV_MNE_COORD_MNI_TAL,
    # ras=FIFF.FIFFV_MNE_COORD_RAS,
    # fs_tal=FIFF.FIFFV_MNE_COORD_FS_TAL,
    # coord_frame = ch_pos.get('coord_frame')
    sensors = Sensors(
        ch_names,
        x,
        y,
        z,
        coord_system=coord_system,
        elecs_fname=elecs_fname,
        coordsystem_fname=coordsystem_fname,
        coord_unit=unit,
        intended_for=intended_img_path,
    )
    return sensors
