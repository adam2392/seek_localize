from pathlib import Path

import pandas as pd
import pytest
from mne.utils import run_tests_if_main
from mne_bids import BIDSPath

from seek_localize import fs_lut_fpath
from seek_localize.label import label_elecs_anat
from seek_localize.bids import bids_validate

# BIDS entities
subject = 'test'
acquisition = 'seeg'
datatype = 'ieeg'
space = 'fs'

# paths to test files
bids_root = Path('data')
# bids_root = Path('/Users/adam2392/Downloads/data/')
deriv_root = bids_root / 'derivatives'
mri_dir = deriv_root / 'freesurfer' / f'sub-{subject}' / 'mri'

desikan_fname = mri_dir / 'aparc+aseg.mgz'
destrieux_fname = mri_dir / 'aparc.a2009s+aseg.mgz'
wmparc_fname = mri_dir / 'wmparc.mgz'
T1mgz = mri_dir / 'T1.mgz'

# path to BIDs electrodes tsv file in test dataset
# NOTE: should not be used directly, always copy to temp directory
_bids_path = BIDSPath(subject=subject,
                      acquisition=acquisition, datatype=datatype,
                      space=space, root=bids_root,
                      suffix='electrodes', extension='.tsv')


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
    bids_path = _bids_path.copy().update(
        root=_temp_bids_root
    )
    coordsystem_fpath = bids_path.copy().update(suffix='coordsystem',
                                                extension='.json')

    # read in the original electrodes file
    elecs_df = pd.read_csv(bids_path.copy(), delimiter='\t', usecols=['name', 'x', 'y', 'z'])

    # import shutil as sh
    # sh.copy(bids_path, bids_path.copy().update(space='testing'))
    # bids_path.update(space='testing')
    # sh.copy(coordsystem_fpath, bids_path.copy().update(suffix='coordsystem',
    #                                             extension='.json'
    # ))
    elecs_df.to_csv(bids_path, sep="\t", index=None)

    # now attempt to label anat
    new_elecs_df = label_elecs_anat(bids_path, img_fname,
                                    fs_lut_fpath=fs_lut_fpath, round=False)
    new_elecs_df.to_csv(bids_path, sep="\t", index=None)

    # original dataframe should not change
    for column in elecs_df.columns:
        pd.testing.assert_series_equal(elecs_df[column], new_elecs_df[column])

    # check that it equals what it was before
    expected_df = pd.read_csv(_bids_path, delimiter='\t', index_col=None)
    if atlas_name in expected_df.columns:
        pd.testing.assert_series_equal(expected_df[atlas_name],
                                       new_elecs_df[atlas_name])

    # new labeled column should exist
    assert atlas_name in new_elecs_df.columns
    assert atlas_name not in elecs_df.columns

    # reset for other tests and fixtures
    elecs_df.to_csv(bids_path, sep='\t', index=None)

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


run_tests_if_main()
