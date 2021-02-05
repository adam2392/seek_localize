from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath
from nibabel.affines import apply_affine

from seek_localize import fs_lut_fpath
from seek_localize.bids import _read_coords_json, read_dig_bids
from seek_localize.label import label_elecs_anat, convert_elecs_coords

# BIDS entities
subject = 'la02'
session = 'presurgery'
acquisition = 'seeg'
datatype = 'ieeg'
space = 'fs'

# paths to test files
bids_root = Path('data')
deriv_root = bids_root / 'derivatives'
mri_dir = deriv_root / 'freesurfer' / f'sub-{subject}' / 'mri'

desikan_fname = mri_dir / 'aparc+aseg.mgz'
destrieux_fname = mri_dir / 'aparc.a2009s+aseg.mgz'
wmparc_fname = mri_dir / 'wmparc.mgz'

# path to BIDs electrodes tsv file in test dataset
# NOTE: should not be used directly, always copy to temp directory
_bids_path = BIDSPath(subject=subject, session=session,
                      acquisition=acquisition, datatype=datatype,
                      space=space, root=bids_root,
                      suffix='electrodes', extension='.tsv')


@pytest.mark.usefixtures('_temp_bids_root')
@pytest.mark.parametrize('to_coord', ['voxel', 'mm', 'm'])
def test_convert_coordunits(_temp_bids_root, to_coord):
    """Test conversion of coordinate units between voxel and xyz."""
    bids_path = _bids_path.copy().update(root=_temp_bids_root)

    coordsystem_fpath = bids_path.copy().update(suffix='coordsystem',
                                                extension='.json')

    # read dig bids
    sensors_mm = read_dig_bids(bids_path, coordsystem_fpath)
    img_fpath = sensors_mm.intended_for

    # check error on unexpected kwarg
    if to_coord == 'm':
        with pytest.raises(ValueError,
                           match='Converting coordinates '
                                 'to m is not accepted.'):
            convert_elecs_coords(sensors=sensors_mm,
                                 to_coord=to_coord)
        return

    # convert sensors
    sensors_conv = convert_elecs_coords(sensors=sensors_mm,
                                        to_coord=to_coord,
                                        round=False)

    # intended for image path should match
    assert img_fpath == sensors_conv.intended_for

    # new coordinate unit should be set
    assert sensors_conv.coord_unit == to_coord

    # convert to voxel
    if to_coord == 'voxel':
        # apply affine yourself to go from mm -> voxels
        img = nb.load(img_fpath)
        inv_affine = np.linalg.inv(img.affine)
        coords = apply_affine(inv_affine, sensors_mm.get_coords().copy())

        # round trip should be the same
        sensors_mm_new = convert_elecs_coords(sensors=sensors_conv,
                                              to_coord='mm',
                                              round=False)
        # the coordinates should match
        np.testing.assert_array_almost_equal(sensors_mm_new.get_coords(),
                                      sensors_mm.get_coords())
    elif to_coord == 'mm':
        coords = sensors_mm.get_coords()


    # the coordinates should match
    np.testing.assert_array_equal(coords, sensors_conv.get_coords())
    assert all([sensors_conv.__dict__[key] == sensors_mm.__dict__[key]
                for key in
                sensors_conv.__dict__.keys() if key not in ['x', 'y', 'z', 'coord_unit']])


@pytest.mark.usefixtures('_temp_bids_root')
@pytest.mark.parametrize('img_fname, atlas_name, expected_anatomy', [
    [desikan_fname, 'desikan-killiany', 'ctx-lh-superiorfrontal'],
    [destrieux_fname, 'destrieux', 'ctx_lh_G_front_sup'],
    [wmparc_fname, 'desikan-killiany-wm', 'ctx-lh-superiorfrontal'],
])
def test_anat_labeling(_temp_bids_root, img_fname, atlas_name, expected_anatomy):
    """Test anatomical labeling of electrodes.tsv files.

    Should work regardless of input coordinate system.
    """
    bids_path = _bids_path.copy().update(root=_temp_bids_root)

    coordsystem_fpath = bids_path.copy().update(suffix='coordsystem',
                                                extension='.json')

    # read in the original electrodes file
    elecs_df = pd.read_csv(bids_path, delimiter='\t')
    coord_system = _read_coords_json(coordsystem_fpath)

    # now attempt to label anat
    label_elecs_anat(bids_path, img_fname, fs_lut_fpath=fs_lut_fpath, round=False)

    # read in the new elecs file
    new_elecs_df = pd.read_csv(bids_path, delimiter='\t')

    # original dataframe should not change
    for column in elecs_df.columns:
        pd.testing.assert_series_equal(elecs_df[column], new_elecs_df[column])

    # new labeled column should exist
    assert atlas_name in new_elecs_df.columns
    assert atlas_name not in elecs_df.columns

    # reset for other tests and fixtures
    elecs_df.to_csv(bids_path, sep='\t', index=None)

    # check labeled anatomy where the lesion was
    idx = new_elecs_df.index[new_elecs_df['name'] == "L'2"]
    assert new_elecs_df[atlas_name][idx].values[0] == expected_anatomy

    # check that it equals what it was before
    expected_df = pd.read_csv(_bids_path, delimiter='\t', index_col=None)
    if atlas_name in expected_df.columns:
        pd.testing.assert_series_equal(expected_df[atlas_name],
                                       new_elecs_df[atlas_name])

    # TODO: MAKE WORK BY COPYING OVER BIDSIGNORE FILE
    # bids_validate(bids_path.root)

    # test errors
    with pytest.raises(ValueError, match='Image must be one of'):
        label_elecs_anat(_bids_path, img_fname='blah.nii',
                         fs_lut_fpath=fs_lut_fpath)

    with pytest.raises(ValueError, match='BIDS path input '
                                         'should lead to '
                                         'the electrodes.tsv'):
        label_elecs_anat(bids_path.copy().update(suffix='ieeg'),
                         img_fname=img_fname,
                         fs_lut_fpath=fs_lut_fpath)
