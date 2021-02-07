"""
.. currentmodule:: seek_localize

.. _label_anatomy_channels-example:

===============================
01. Label Anatomy of Electrodes
===============================

When working with intracranial electrophysiological data in the
BIDS format, we usually have

- iEEG (ECoG and SEEG)
- the anatomical MRI scan of a study participant
- the CT scan of the study participant with iEEG electrodes implanted

In this tutorial, we show how ``label_elecs_anat`` can be used to
quickly and easily label anatomy of electrodes.

We assume that you have already localized the electrodes and coregistered
them over to the T1w image FreeSurfer space.

"""  # noqa: E501

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# Step 1: Imports
# ---------------
# We are importing everything we need for this example:
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath, print_dir_tree, make_report

from seek_localize import label_elecs_anat, fs_lut_fpath

###############################################################################
# We will be using the `testing dataset`, which
# is already stored in BIDS format and stored with the
# ``seek-localize`` repository.

bids_root = Path.cwd() / Path("../data/")
fs_root = bids_root / "derivatives" / "freesurfer"

###############################################################################
# Step 2: Explore the dataset contents
# ------------------------------------
#
# We can use MNE-BIDS to print a tree of all
# included files and folders. We pass the ``max_depth`` parameter to
# `mne_bids.print_dir_tree` to the output to three levels of folders, for
# better readability in this example.

print_dir_tree(bids_root, max_depth=3)

###############################################################################
# We can even ask MNE-BIDS to produce a human-readable summary report
# on the dataset contents.

print(make_report(bids_root))

###############################################################################
# Step 3: Label the anatomy of electrodes
# ---------------------------------------
# Now it's time to get ready for labeling some of the data! First, we need to
# create a :func:`mne_bids.BIDSPath`, which will point to the corresponding
# ``*electrodes.tsv`` file.
#

subject = "la02"
session = "presurgery"
acquisition = "seeg"
space = "fs"
suffix = "electrodes"
extension = ".tsv"
datatype = "ieeg"
bids_path = BIDSPath(
    root=bids_root,
    datatype=datatype,
    subject=subject,
    session=session,
    acquisition=acquisition,
    space=space,
    suffix=suffix,
    extension=extension,
)

# the full file path to the electrodes.tsv file
print(bids_path.fpath)

###############################################################################
# The necessary iEEG files are the
# ``sub-la02_ses-presurgery_acq-seeg_space-fs_electrodes.tsv``,
# ``sub-la02_ses-presurgery_acq-seeg_space-fs_coordsystem.json`` files. Note
# these are co-occurring files in iEEG-BIDS_ (one present requires the other to
# be present).
#

coordsystem_fpath = bids_path.copy().update(suffix="coordsystem", extension=".json")
print(coordsystem_fpath.fpath)

###############################################################################
# Let's explore the contents of the current electrodes.tsv file.
# Note that the current data already has the atlas labels, so we
# pretend they are not there and only read in the bare minimum columns.
#

elec_df = pd.read_csv(
    bids_path, sep="\t", index_col=None, usecols=["name", "x", "y", "z"]
)
print(elec_df)

###############################################################################
# The necessary imaging files are the
# ``sub-la02_ses-presurgery_space-fs_T1w.nii`` file, which the electrode
# coordinates are assumed to be in.
#

atlas_img_fpath = fs_root / f"sub-{subject}" / "mri" / "aparc+aseg.mgz"

###############################################################################
# Now let's label the anatomy!
# Note: seek_localize.fs_lut_fpath provides the file path to a local
# ``FreeSurferColorLUT.txt`` file.

elec_df = label_elecs_anat(bids_path, atlas_img_fpath, fs_lut_fpath=fs_lut_fpath)

print(elec_df)

###############################################################################
# .. LINKS
#
# .. _iEEG-BIDS:
#    https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/04-intracranial-electroencephalography.html
