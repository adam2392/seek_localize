import os
from pathlib import Path

from mne_bids import BIDSPath

from seek_localize import read_dig_bids
from seek_localize.bids import write_dig_bids

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


def test_bids_write(_temp_bids_root):
    """Test BIDS writing and reading.

    Test that electrodes.tsv and coordsystem.json files writing
    to BIDS is i) BIDS compliant and ii) readable by mne-bids again.
    """
    bids_path = _bids_path.copy().update(root=_temp_bids_root)

    sensors = read_dig_bids(bids_path, root=_temp_bids_root)

    ch_names = sensors.ch_names
    ch_coords = sensors.get_coords()
    unit = sensors.coord_unit
    coord_system = sensors.coord_system
    intended_for = sensors.intended_for

    elec_bids_path = BIDSPath(subject='02',
                              session=bids_path.session,
                              space=bids_path.space,
                              acquisition=bids_path.acquisition,
                              datatype=bids_path.datatype,
                              root=_temp_bids_root,
                              suffix='electrodes',
                              extension='.tsv')

    write_dig_bids(elec_bids_path, root=_temp_bids_root,
                   ch_names=ch_names,
                   ch_coords=ch_coords, unit=unit,
                   coord_system=coord_system,
                   intended_for=intended_for)
    new_sensors = read_dig_bids(elec_bids_path, root=_temp_bids_root)

    # the coordinates should match
    assert all([sensors.__dict__[key] == new_sensors.__dict__[key]
                for key in
                sensors.__dict__.keys() if key not in [
                    'elecs_fname', 'coordsystem_fname'
                ]])
