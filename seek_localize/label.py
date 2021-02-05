import os
from pathlib import Path
from typing import Union, Dict

import nibabel as nb
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from nibabel.affines import apply_affine

from seek_localize.bids import read_dig_bids, Sensors, _match_dig_sidecars
from seek_localize.config import MAPPING_COORD_FRAMES, ACCEPTED_IMAGE_VOLUMES


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
            lab[np.int(row[0])] = lname

    return lab


def convert_elecs_coords(sensors: Sensors, to_coord: str):
    """Convert electrode coordinates between voxel and xyz.

    To obtain the sensors, one can use :func:`seek_localize.bids.read_dig_bids`.

    Parameters
    ----------
    sensors : Sensors
        An instance of the electrode sensors with the coordinates,
        coordinate system and coordinate units.
    to_coord : str
        The type of coordinate unit to convert to. Must be one of
        ``['voxel', 'mm']``.
    img : instance of Nifti image | None
        Image volume that sensors are intended for that supplies the
        affine transformation to go from 'mm' <-> 'voxels'.

    Returns
    -------
    sensors : Sensors
        The electrode sensors with converted coordinates.
    """
    if to_coord not in MAPPING_COORD_FRAMES:
        raise ValueError(
            f"Converting coordinates to {to_coord} "
            f"is not accepted. Please use one of "
            f"{MAPPING_COORD_FRAMES} coordinate systems."
        )

    # get the image file path
    img_fpath = sensors.intended_for
    img = nb.load(img_fpath)

    # voxel -> xyz
    affine = img.affine

    # xyz -> voxel
    inv_affine = np.linalg.inv(affine)

    # get the actual xyz coordinates
    elec_coords = sensors.get_coords()

    print(f"Got electrode coordinates {elec_coords.shape}")

    # apply the affine
    if to_coord == "voxel":
        elec_coords = apply_affine(inv_affine, elec_coords)
    elif to_coord == "mm":
        elec_coords = apply_affine(affine, elec_coords)

        # convert from m to mm
        elec_coords = np.divide(elec_coords, 1000.0)

    # recreate sensors
    sensors = Sensors(**sensors.__dict__)
    sensors.set_coords(elec_coords)
    sensors.coord_unit = to_coord
    return sensors


def label_elecs_anat(
    bids_path: BIDSPath,
    img_fname: Union[str, Path],
    fs_lut_fpath: Union[str, Path],
    verbose: bool = True,
):
    """Label electrode anatomical location based on an annotated image volume.

    Parameters
    ----------
    bids_path : BIDSPath
        The BIDS path constructed using :func:`mne_bids.BIDSPath` that leads
        to the ``*electrodes.tsv`` file.
    img_fname : str | pathlib.Path
        The file path for the image volume. Must be a Nifti file.
    fs_lut_fpath : str | pathlib.Path
        The file path for the ``FreeSurferColorLUT.txt`` file.
    verbose : bool
        Verbosity.
    """
    # work with pathlib
    img_fname = Path(img_fname)

    # error check atlas image volume
    if os.path.basename(img_fname) not in ACCEPTED_IMAGE_VOLUMES:
        raise ValueError(
            "Image must be one of FreeSurfer "
            "output annotated image volumes: "
            f"{ACCEPTED_IMAGE_VOLUMES}."
        )

    if bids_path.suffix != "electrodes" or bids_path.extension != ".tsv":
        raise ValueError(
            f"BIDS path input should lead "
            f"to the electrodes.tsv file. "
            f"{bids_path} does not have "
            f"suffix electrodes, and/or "
            f"extension .tsv."
        )

    if "aparc.a2009s+aseg.mgz" == img_fname.name:
        atlas_name = "destrieux"
    elif "aparc+aseg.mgz" == img_fname.name:
        atlas_name = "desikan-killiany"
    elif "wmparc.mgz" == img_fname.name:
        atlas_name = "desikan-killiany-wm"

    # read in the atlas volume image
    atlas_img = nb.freesurfer.load(img_fname)

    # get the names of these labels using Freesurfer's lookup table (LUT)
    if verbose:
        print(f"Loading lookup table for freesurfer labels: {fs_lut_fpath}")
    lut_labels = _read_lut_file(lut_fname=fs_lut_fpath)

    # get the paths to the electrodes/coordsystem.json files
    elecs_path, coordsystem_path = _match_dig_sidecars(bids_path)
    coordsystem_fname = coordsystem_path.fpath
    elecs_fname = elecs_path.fpath

    # read in the electrodes / coordsystem
    elecs = read_dig_bids(elecs_fname, coordsystem_fname=coordsystem_fname)

    # convert elecs to voxel coordinates
    if elecs.coord_unit != "voxel":
        elecs = convert_elecs_coords(sensors=elecs, to_coord="voxel")

    # map wrt atlas
    elec_coords = elecs.get_coords()
    anatomy_labels = _label_depth(
        elec_coords, atlas_img=atlas_img, lut_labels=lut_labels
    )

    # first get all the coordinates as a dictionary
    elecs_dict = elecs.as_dict()
    elecs_dict[atlas_name] = anatomy_labels

    # update electrodes.tsv file with anatomical labels
    orig_elecs_tsv = pd.read_csv(elecs_path, delimiter="\t")
    orig_elecs_tsv[atlas_name] = anatomy_labels
    orig_elecs_tsv.to_csv(elecs_path, sep="\t", index=None)


def _label_depth(
    elec_coords: np.ndarray,
    atlas_img: nb.Nifti2Image,
    lut_labels: Dict,
    verbose: bool = True,
):
    """Label depth electrode voxel coordinates in the atlas image space.

    Requires the electrode coordinates to already be in voxel space.

    Parameters
    ----------
    elec_coords : np.ndarray (n_channels, 3)
        The voxel coordinates for the electrodes.
    atlas_img : nb.Nifti2Image
        The corresponding Nifti volumetric image that has an atlas
        segmentation at the voxel level.
    lut_labels : dict
        A dictionary of voxel coordinates that are mapped to the
        atlas anatomical label.
    verbose : bool
        Verbosity

    Returns
    -------
    anatomy_labels : List
        A list of the anatomical labels per each of the electrode voxel coordinates.

    Notes
    -----
    The ``tkrvox2ras`` transformation can be obtained from FreeSurfer's ``mri_info``
    command via::

        mri_info --vox2ras-tkr <img>

    This will generally be the 4x4 matrix for FreeSurfer output.::

            [
                [-1.0, 0.0, 0.0, 128.0],
                [0.0, 0.0, 1.0, -128.0],
                [0.0, -1.0, 0.0, 128.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
    """
    # affine transformation
    # Define the affine transform to go from surface coordinates to volume coordinates (as CRS, which is
    # the slice *number* as x,y,z in the 3D volume. That is, if there are 256 x 256 x 256 voxels, the
    # CRS coordinate will go from 0 to 255.)
    vox2ras_affine = np.array(
        [
            [-1.0, 0.0, 0.0, 128.0],
            [0.0, 0.0, 1.0, -128.0],
            [0.0, -1.0, 0.0, 128.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # get the dummy 4th dimension for the 3D coordinates
    # to then apply an affine transformation to
    intercept = np.ones(len(elec_coords))
    elecs_ones = np.column_stack((elec_coords, intercept))

    # find voxel CRS
    # apply inverse of tkrvox2ras
    inv_affine = np.linalg.inv(vox2ras_affine)
    VoxCRS = np.dot(inv_affine, elecs_ones.transpose()).transpose().astype(int)

    if verbose:
        print("Labeling electrodes...")
        print(
            "VoxCRS limits in 3D: ",
            np.max(VoxCRS[:, 0]),
            np.max(VoxCRS[:, 1]),
            np.max(VoxCRS[:, 2]),
        )

    # Label the electrodes according to the aseg volume
    nchans = elec_coords.shape[0]

    # get the atlas data
    aparc_dat = atlas_img.get_fdata()

    # label each channel
    anatomy_labels = []
    for idx in range(nchans):
        voxel = elec_coords[idx, :].astype(
            int
        )  # VoxCRS[elec, 0], VoxCRS[elec, 1], VoxCRS[elec, 2]
        voxel_num = aparc_dat[voxel[0], voxel[1], voxel[2]]
        label = lut_labels[voxel_num]
        anatomy_labels.append(label)
        if verbose:
            print(f"E{idx}, Vox CRS: {voxel}, Label #{label}")

    return anatomy_labels
