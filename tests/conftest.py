"""Define fixtures available for eztrack testing."""
import os.path
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mne.utils import run_subprocess
from mne_bids import BIDSPath

np.random.seed(987654321)

# bids root - testing data
BIDS_ROOT = Path('data')


# WINDOWS issues:
# the bids-validator development version does not work properly on Windows as
# of 2019-06-25 --> https://github.com/bids-standard/bids-validator/issues/790
# As a workaround, we try to get the path to the executable from an environment
# variable VALIDATOR_EXECUTABLE ... if this is not possible we assume to be
# using the stable bids-validator and make a direct call of bids-validator
# also: for windows, shell = True is needed to call npm, bids-validator etc.
# see: https://stackoverflow.com/q/28891053/5201771
@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""
    shell = False
    bids_validator_exe = ["bids-validator", "--config.error=41", "--config.error=41"]
    if platform.system() == "Windows":
        shell = True
        exe = os.getenv("VALIDATOR_EXECUTABLE", "n/a")
        if "VALIDATOR_EXECUTABLE" != "n/a":
            bids_validator_exe = ["node", exe]

    def _validate(bids_root):
        try:
            cmd = bids_validator_exe + [bids_root]
            run_subprocess(cmd, shell=shell)
        except Exception as e:
            print(e)
            return True

    return _validate


@pytest.fixture(scope='function')
def _temp_bids_root(tmpdir):
    # paths to test files
    bids_root = Path(os.getcwd()) / 'data'

    # comment out when not running locally
    # tmpdir = '/Users/adam2392/Downloads/data/'
    # bids_root = Path('/Users/adam2392/Documents/seek_localize/data')

    # shutil copy entire thing
    shutil.copytree(bids_root, tmpdir, dirs_exist_ok=True)

    # BIDS entities
    for subject in ['la02', 'test']:
        session = 'presurgery'
        acquisition = 'seeg'
        datatype = 'ieeg'
        space = 'fs'

        # path to BIDs electrodes tsv file in test dataset
        # NOTE: should not be used directly, always copy to temp directory
        _bids_path = BIDSPath(subject=subject, session=session,
                              acquisition=acquisition, datatype=datatype,
                              space=space, root=tmpdir,
                              suffix='electrodes', extension='.tsv')

        if subject == 'test':
            _bids_path.update(session=None)

        # read in the original electrodes file
        elecs_df = pd.read_csv(_bids_path, delimiter='\t', index_col=None)
        labels = ['destrieux', 'desikan-killiany', 'desikan-killiany-wm']
        elecs_df.drop(labels=labels,
                      axis=1,
                      inplace=True,
                      errors='ignore')
        elecs_df.to_csv(_bids_path, sep='\t', index=None)
    return tmpdir
