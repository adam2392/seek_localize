import os
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from mne import read_talxfm
from mne_bids import BIDSPath
from nibabel.affines import apply_affine

from seek_localize import read_dig_bids, convert_coord_units, convert_coord_space

# BIDS entities
subject = 'la02'
session = 'presurgery'
acquisition = 'seeg'
datatype = 'ieeg'
space = 'fs'

# paths to test files
cwd = os.getcwd()
bids_root = Path(cwd) / 'data'
deriv_root = bids_root / 'derivatives'
mri_dir = deriv_root / 'freesurfer' / f'sub-{subject}' / 'mri'
subjects_dir = deriv_root / 'freesurfer'

desikan_fname = mri_dir / 'aparc+aseg.mgz'
destrieux_fname = mri_dir / 'aparc.a2009s+aseg.mgz'
wmparc_fname = mri_dir / 'wmparc.mgz'
T1mgz = mri_dir / 'T1.mgz'

# path to BIDs electrodes tsv file in test dataset
# NOTE: should not be used directly, always copy to temp directory
_bids_path = BIDSPath(subject=subject, session=session,
                      acquisition=acquisition, datatype=datatype,
                      space=space, root=bids_root,
                      suffix='electrodes', extension='.tsv')


@pytest.mark.usefixtures('_temp_bids_root')
@pytest.mark.parametrize('to_frame, to_unit', [
    ['mri', 'm'],
    ['mri', 'voxel'],
    ['mri', 'mm'],
    ['tkras', 'mm'],
    ['mni', 'voxel'],
    ['mni', 'mm'],
    ['tkras', 'voxel'],
])
def test_convert_coordunits(_temp_bids_root, to_frame, to_unit):
    """Test conversion of coordinate units between voxel and xyz."""
    bids_path = _bids_path.copy().update(root=_temp_bids_root)
    subject = bids_path.subject
    coordsystem_fpath = bids_path.copy().update(suffix='coordsystem',
                                                extension='.json')

    # read dig bids
    sensors_mm = read_dig_bids(bids_path, root=_temp_bids_root)
    img_fpath = sensors_mm.intended_for

    # check error on unexpected kwarg
    if to_unit == 'm':
        sensors_m = convert_coord_units(sensors=sensors_mm,
                                        to_unit=to_unit,
                                        round=False)
        np.testing.assert_array_almost_equal(
            1000. * sensors_m.get_coords(),
            sensors_mm.get_coords())
        with pytest.raises(ValueError,
                           match='Converting coordinates '
                                 'to km is not accepted.'):
            convert_coord_units(sensors=sensors_mm,
                                to_unit='km')
        return
    elif to_frame in ['tkras', 'mni']:
        # conversion should not work if starting from xyz coords
        with pytest.raises(ValueError,
                           match='Converting coordinates '
                                 'to .* is not accepted.'):
            convert_coord_units(sensors=sensors_mm,
                                to_unit=to_frame)

        # conversion should not work if trying to go to mm
        with pytest.raises(ValueError,
                           match='Converting coordinates '
                                 'to mm is not accepted.'):
            convert_coord_space(sensors=sensors_mm,
                                to_frame='mm')

        # conversion should not work if trying to start
        # from mm xyz coords
        with pytest.raises(ValueError,
                           match='Converting coordinates '
                                 'requires sensor'):
            convert_coord_space(sensors=sensors_mm,
                                to_frame='tkras')

    # default settings for the read in coordinates file
    assert sensors_mm.coord_unit == 'mm'
    assert sensors_mm.coord_system == 'mri'

    # test units
    if to_unit == 'mm':
        # convert sensors
        sensors_conv = convert_coord_units(sensors=sensors_mm,
                                           to_unit=to_unit,
                                           round=False)
        np.testing.assert_array_almost_equal(sensors_conv.get_coords(),
                                             sensors_mm.get_coords())
    elif to_unit == 'voxel':
        pass

    # convert to voxel should just require convert elec coords
    if to_frame == 'mri':
        # convert sensors first to voxels
        sensors_conv = convert_coord_units(sensors=sensors_mm,
                                           to_unit='voxel',
                                           round=False)

        # returning nothing should be the same
        sensors_mm_mri = convert_coord_space(sensors_mm,
                                             to_frame=to_frame)

        # apply affine yourself to go from mm -> voxels
        img = nb.load(img_fpath)
        inv_affine = np.linalg.inv(img.affine)
        coords = apply_affine(inv_affine, sensors_mm.get_coords().copy())

        # round trip should be the same
        sensors_mm_new = convert_coord_units(sensors=sensors_conv,
                                             to_unit='mm',
                                             round=False)
        # the coordinates should match
        np.testing.assert_array_almost_equal(sensors_mm_new.get_coords(),
                                             sensors_mm.get_coords())

    elif to_frame == 'mni':
        # convert to voxels
        sensors_vox = convert_coord_units(sensors=sensors_mm,
                                          to_unit='voxel',
                                          round=False)
        # convert voxels to mni
        sensors_conv = convert_coord_space(sensors_vox, to_frame=to_frame,
                                           subjects_dir=subjects_dir)

        # load FreeSurfer -> MNI transform (i.e. fsaverage)
        mni_mri_t = read_talxfm(subject=f'sub-{subject}',
                                subjects_dir=subjects_dir)

        # go voxels -> tkras
        coords = apply_affine(mni_mri_t['trans'] * 1000.,
                              sensors_vox.get_coords())

        # the coordinates should match
        np.testing.assert_array_almost_equal(sensors_conv.get_coords(),
                                             coords)
    elif to_frame == 'tkras':
        # convert to voxels
        sensors_vox = convert_coord_units(sensors=sensors_mm,
                                          to_unit='voxel',
                                          round=False)
        # convert voxels to tkras
        sensors_conv = convert_coord_space(sensors_vox, to_frame=to_frame)

        # load FreeSurfer MGH file
        img = nb.load(T1mgz)

        # go voxels -> tkras
        coords = apply_affine(img.header.get_vox2ras_tkr(), sensors_vox.get_coords())

        # the coordinates should match
        np.testing.assert_array_almost_equal(sensors_conv.get_coords(),
                                             coords)

    # intended for image path should match
    assert img_fpath == sensors_conv.intended_for

    # new coordinate unit should be set
    if to_frame == 'tkras':
        sensors_conv.coord_unit == 'mm'
    else:
        assert sensors_conv.coord_unit == 'voxel'
    assert sensors_conv.coord_system == to_frame

    # the coordinates should match
    np.testing.assert_array_equal(coords, sensors_conv.get_coords())
    assert all([sensors_conv.__dict__[key] == sensors_mm.__dict__[key]
                for key in
                sensors_conv.__dict__.keys() if key not in ['x', 'y', 'z',
                                                            'coord_unit',
                                                            'coord_system']])
