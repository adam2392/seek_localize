# possible coordinate frames
MAPPING_COORD_FRAMES = ["mri", "tkras", "fsaverage"]

# SI units for coordinates
SI = {"mm": 0.001, "cm": 0.01, "m": 1.0}

# possible coordinate units
COORDINATE_UNITS = ["voxel"]
COORDINATE_UNITS.extend(list(SI.keys()))

ACCEPTED_IMAGE_VOLUMES = [
    "wmparc.mgz",  # wm parcellation
    "aparc.a2009s+aseg.mgz",  # destrieux
    "aparc+aseg.mgz",  # desikan-killiany
]
ACCEPTED_MNE_COORD_FRAMES = ["mri", "mri_voxel", "mni_tal", "ras", "fs_tal"]

COORD_FRAME_DESCRIPTIONS = {
    'individual': 'Participant specific anatomical space (for example '
                  'derived from T1w and/or T2w images). This coordinate '
                  'system requires specifying an additional, '
                  'participant-specific file to be fully defined.',
    'fsaverage': 'Freesurfer average template (in MNI305 space) '
                 'coordinates',
    'ras': 'RAS means that the first dimension (X) points towards '
           'the right hand side of the head, the second dimension (Y) '
           'points towards the Anterior aspect of the head, and the '
           'third dimension (Z) points towards the top of the head.',
}