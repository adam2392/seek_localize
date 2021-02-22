"""
.. currentmodule:: seek_localize

.. _convert_channel_coordframes-example:

===========================================
02. Convert Coordinate Frames of Electrodes
===========================================

When working with intracranial electrophysiological data in the
iEEG-BIDS_ format, we usually have iEEG coordinate data either in
``voxel``, or real world coordinates space (xyz coordinates).
Then within xyz coordinates, it can either be ``RAS``, or
``tkRAS`` if one uses FreeSurfer.

In this tutorial, we show how to quickly use the ``Sensors``
data class and quickly go back and forth between coordinate frames
using ``convert_elec_coords``.

We assume that you have already localized the electrodes and coregistered
them over to the T1w image FreeSurfer space.

"""  # noqa: E501

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# Imports
# -------
# We are importing everything we need for this example:
from pathlib import Path

from mne_bids import BIDSPath

from seek_localize import read_dig_bids, convert_coord_units, convert_coord_space

###############################################################################
# We will be using the `testing dataset`, which
# is already stored in BIDS format and stored with the
# ``seek-localize`` repository.

bids_root = (Path.cwd() / Path("../data/")).absolute()
subjects_dir = bids_root / "derivatives" / "freesurfer"

###############################################################################
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
electrodes_fpath = BIDSPath(
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
print(electrodes_fpath.fpath)

###############################################################################
# The necessary iEEG files are the
# ``sub-la02_ses-presurgery_acq-seeg_space-fs_electrodes.tsv``,
# ``sub-la02_ses-presurgery_acq-seeg_space-fs_coordsystem.json`` files. Note
# these are co-occurring files in iEEG-BIDS_ (one present requires the other to
# be present).
#

coordsystem_fpath = electrodes_fpath.copy().update(
    suffix="coordsystem", extension=".json"
)
print(coordsystem_fpath.fpath)

###############################################################################
# Let's load in the electrode coordinates as an instance of the
# `seek_localize.Sensors` class. Rather then instantiating the class
# directly, we use `seek_localize.read_dig_bids` to read in the
# correct data. This will perform extra work, such as figuring
# out the full path to the ``IntendedFor`` volumetric image. The
# image corresponds to the coordinate space to interpret the
# electrode coordinates in (e.g. a T1w image in FreeSurfer space).
#

sensors = read_dig_bids(electrodes_fpath, root=bids_root)
print(sensors)

###############################################################################
# The data already saved was originally written in ``'mm'``, so we can
# convert to ``voxel`` space denoted by the ``mri`` coordinate frame.
# This is in-line with how MNE_ does things
#

sensors_vox = convert_coord_units(sensors, to_unit="voxel")
print(sensors_vox)

###############################################################################
# We could convert it to ``mm``.
sensors_mm = convert_coord_units(sensors_vox, to_unit="mm")
print(sensors_mm)

###############################################################################
# The data was originally saved according to the ``mri`` space, intended
# for the ``T1.mgz`` image in FreeSurfer. One can also use seek_localize to
# transform to standard coordinate spaces, such as ``tkras`` and ``mni``.
#

# We could convert it to ``tkras``.
sensors_tkras = convert_coord_space(sensors_vox, to_frame="tkras")
print(sensors_tkras)

# We could convert it to ``mni``.
sensors_mni = convert_coord_space(
    sensors_vox, to_frame="mni", subjects_dir=subjects_dir
)
print(sensors_mni)

# We could convert it to back to ``mri``.
sensors_mri = convert_coord_space(sensors_vox, to_frame="mri")
print(sensors_mri)

###############################################################################
# .. LINKS
#
# .. _iEEG-BIDS:
#    https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/04-intracranial-electroencephalography.html
# .. _MNE: https://mne.tools/dev/auto_tutorials/source-modeling/plot_source_alignment.html
