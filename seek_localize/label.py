import os
from pathlib import Path
from typing import Union, Dict, Any

import nibabel as nb
import numpy as np
import pandas as pd
from mne.utils import warn
from mne_bids import BIDSPath
from nibabel.affines import apply_affine
from nptyping import NDArray

from seek_localize.bids import _match_dig_sidecars
from seek_localize.config import MAPPING_COORD_FRAMES, ACCEPTED_IMAGE_VOLUMES
from seek_localize.electrodes import Sensors
from seek_localize.io import read_dig_bids, _read_lut_file
from seek_localize.utils import (
    _scale_coordinates,
    _read_vertex_labels,
    _read_cortex_vertices,
)
from seek_localize.utils.space import nearest_electrode_vert


def convert_elecs_coords(sensors: Sensors, to_coord: str, round=True):
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
    round : bool
        Whether to round the coordinates to the nearest integer.
    img : instance of Nifti image | None
        Image volume that sensors are intended for that supplies the
        affine transformation to go from 'mm' <-> 'voxels'.

    Returns
    -------
    sensors : Sensors
        The electrode sensors with converted coordinates.

    Notes
    -----
    ``Nibabel`` processes everything in units of ``millimeters``.

    To convert from xyz (e.g. 'mm') to voxel and vice versa, one
    simply needs the ``IntendedFor`` image that contains the affine
    ``vox2ras`` transformation. For example, this might be a T1w
    image. One can use :func:`nibabel.affines.apply_affine` to then
    apply the corresponding transformation from vox to xyz space.

    Note, if you want to go from xyz to vox, then you need the
    inverse of the ``vox2ras`` transformation.

    If one wants to convert to ``tkras``, which is FreeSurfer's
    surface xyz space, this is the xyz space of the closest surface [1,2,3].
    This corresponds to the `vox2rask_tkr <https://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.mghformat.MGHHeader.get_vox2ras_tkr>`_  # noqa
    function in ``nibabel``. The ``tkrvox2ras`` transformation can
    be obtained from FreeSurfer's ``mri_info`` command via::

        mri_info --vox2ras-tkr <img>

    This will generally be the 4x4 matrix for FreeSurfer output.::

            [
                [-1.0, 0.0, 0.0, 128.0],
                [0.0, 0.0, 1.0, -128.0],
                [0.0, -1.0, 0.0, 128.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

    but may be different depending on how some FreeSurfer hyperparameters.

    References
    ----------
    .. [1] FieldTrip explanation: https://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined/#details-of-the-freesurfer-coordinate-system  # noqa

    .. [2] How MNE handles FreeSurfer data: https://mne.tools/dev/auto_tutorials/source-modeling/plot_background_freesurfer_mne.html  # noqa

    .. [3] FreeSurfer Wiki: https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems  # noqa
    """
    if to_coord not in MAPPING_COORD_FRAMES:
        raise ValueError(
            f"Converting coordinates to {to_coord} "
            f"is not accepted. Please use one of "
            f"{MAPPING_COORD_FRAMES} coordinate systems."
        )

    if to_coord == sensors.coord_unit:
        return sensors

    # get the image file path
    img_fpath = sensors.intended_for
    img = nb.load(img_fpath)

    # voxel -> xyz
    affine = img.affine

    # xyz -> voxel
    inv_affine = np.linalg.inv(affine)

    # get the actual xyz coordinates
    elec_coords = sensors.get_coords()

    # apply the affine
    if to_coord == "voxel":
        # first scale to millimeters if not already there
        elec_coords = _scale_coordinates(elec_coords, sensors.coord_unit, "mm")

        # now convert xyz to voxels
        elec_coords = apply_affine(inv_affine, elec_coords)
    elif to_coord == "mm":
        # xyz -> voxels
        elec_coords = apply_affine(affine, elec_coords)
    elif to_coord == "tkras":
        # get the voxel to tkRAS transform
        try:
            vox2ras_tkr = img.header.get_vox2ras_tkr()
        except AttributeError as e:
            warn(
                f"Unable to programmatically get vox2ras TKR, "
                f"so setting manually. "
                f"Error: {e}"
            )
            vox2ras_tkr = [
                [-1.0, 0.0, 0.0, 128.0],
                [0.0, 0.0, 1.0, -128.0],
                [0.0, -1.0, 0.0, 128.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

        if sensors.coord_unit not in ["voxel"]:
            # first scale to millimeters if not already there
            elec_coords = _scale_coordinates(elec_coords, sensors.coord_unit, "mm")

        # now convert xyz to voxels
        elec_coords = apply_affine(inv_affine, elec_coords)

        # now convert voxels to tkras
        elec_coords = apply_affine(vox2ras_tkr, elec_coords)

    if round:
        # round it off to integer
        elec_coords = np.round(elec_coords).astype(int)

    # recreate sensors
    sensors = Sensors(**sensors.__dict__)
    sensors.set_coords(elec_coords)
    sensors.coord_unit = to_coord
    return sensors


def _label_ecog(
    elec_coords: NDArray[Any, 3, Any], fs_subj_dir: Union[str, Path], verbose: bool
):
    """Label ecog electrode voxel coordinates in the atlas image space.

    Requires the electrode coordinates to already be in voxel space.

    Parameters
    ----------
    elec_coords : np.ndarray (n_channels, 3)
        The voxel coordinates for the electrodes.
    fs_subj_dir : str | pathlib.Path
    verbose : bool
        Verbosity

    Returns
    -------
    anatomy_labels : List
        A list of the anatomical labels per each of the electrode voxel coordinates.
    """
    if verbose:
        print("Finding nearest mesh vertex for each electrode")

    fs_subj_dir = Path(fs_subj_dir)
    gyri_dir = fs_subj_dir / "label" / "gyri"
    mesh_dir = fs_subj_dir / "Meshes"

    # read the vertex labels of the surface gyri
    rh_vert_labels = _read_vertex_labels(gyri_labels_dir=gyri_dir, hem="rh")
    lh_vert_labels = _read_vertex_labels(gyri_labels_dir=gyri_dir, hem="lh")

    # read cortex's vertices as a numpy array
    rh_cortex_verts = _read_cortex_vertices(mesh_dir=mesh_dir, hem="rh")
    lh_cortex_verts = _read_cortex_vertices(mesh_dir=mesh_dir, hem="lh")

    # find closest vertex
    rh_vert_inds, rh_nearest_verts, rh_dist_matrix = nearest_electrode_vert(
        rh_cortex_verts, elec_coords, return_dist=True
    )
    lh_vert_inds, lh_nearest_verts, lh_dist_matrix = nearest_electrode_vert(
        lh_cortex_verts, elec_coords, return_dist=True
    )

    assert len(rh_vert_inds) == len(lh_vert_inds)
    n_chans = len(rh_vert_inds)

    ## Now make a dictionary of the label for each electrode
    anatomy_labels = []
    for idx in range(n_chans):
        rh_dists = rh_dist_matrix[idx, :]
        lh_dists = lh_dist_matrix[idx, :]

        # for this electrode, determine the closest hemisphere pial surface
        if rh_dists.min() < lh_dists.min():
            vert_inds = rh_vert_inds
            vert_label = rh_vert_labels
        else:
            vert_inds = lh_vert_inds
            vert_label = lh_vert_labels

        # obtain the anatomical label for the closest pial vertex
        if vert_inds[idx] in vert_label:
            anatomy_labels.append(vert_label[vert_inds[idx]].strip())
        else:
            anatomy_labels.append("Unknown")

    return anatomy_labels


def label_elecs_anat(
    bids_path: BIDSPath,
    img_fname: Union[str, Path],
    fs_lut_fpath: Union[str, Path],
    verbose: bool = True,
    **kwargs,
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
    kwargs : dict
        Keyword arguments to be passed to
        :func:`seek_localize.convert_elec_coords`.

    Returns
    -------
    elecs_df : pd.DataFrame
        DataFrame for electrodes.tsv file. If you would like to save it to
        disc. Suggested code

            elecs_df.to_csv(bids_path, sep='\t', index=None)
    """
    # work with pathlib
    img_fname = Path(img_fname)

    # error check atlas image volume
    if os.path.basename(img_fname) not in ACCEPTED_IMAGE_VOLUMES:
        raise ValueError(
            "Image must be one of FreeSurfer "
            "output annotated image volumes: "
            f"{ACCEPTED_IMAGE_VOLUMES}, not {img_fname}."
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
    print(elecs)

    # convert elecs to voxel coordinates
    if elecs.coord_unit != "voxel":
        if verbose:
            print(
                "Converting to voxel space because electrodes "
                f"are in {elecs.coord_unit} space."
            )
        elecs = convert_elecs_coords(sensors=elecs, to_coord="voxel", **kwargs)

    # map wrt atlas
    elec_coords = elecs.get_coords()
    anatomy_labels = _label_depth(
        elec_coords, atlas_img=atlas_img, lut_labels=lut_labels
    )

    # first get all the coordinates as a dictionary
    elecs_dict = elecs.as_dict()
    elecs_dict[atlas_name] = anatomy_labels

    # update electrodes.tsv file with anatomical labels
    elecs_df = pd.read_csv(elecs_path, delimiter="\t")
    elecs_df[atlas_name] = anatomy_labels

    return elecs_df


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
    """
    if verbose:
        print("Labeling electrodes...")

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
