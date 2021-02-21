from pathlib import Path

from seek_localize.utils import read_fieldtrip_elecs

# BIDS entities
subject = 'la02'
session = 'presurgery'
acquisition = 'seeg'
datatype = 'ieeg'
space = 'fs'

# paths to test files
bids_root = Path('data')
deriv_root = bids_root / 'derivatives'
ft_fpath = deriv_root / 'fieldtrip' / 'stolk' / f'{subject}_elec_acpc_f_al.mat'


def test_read_fieldtrip_output():
    elecs_dict = read_fieldtrip_elecs(ft_fpath)

    # assert all([key in elecs_dict for key in
    #             ['elecmatrix', 'eleclabels']])
