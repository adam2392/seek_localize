# possible coordinate frames
MAPPING_COORD_FRAMES = ["mri", "tkras", "mni"]

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
