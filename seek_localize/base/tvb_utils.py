import os
import subprocess
import tempfile

import nibabel
import numpy as np

from seek_localize.utils.utils import extract_vector


def pial_to_verts_and_triangs(pial_surf) -> (np.ndarray, np.ndarray):
    """Convert pial surface file to vertices and triangles.

    Parameters
    ----------
    pial_surf : str | pathlib.Path
        The file path to the pial surface files (e.g. ``lh.pial``, or ``rh.pial``).

    Returns
    -------
    vertices : np.ndarray
        The N x 3 array containing all vertices and their xyz points.
    triangles : np.ndarray
        The N x 3 array containing all indices of the vertices that comprise
        each triangle.
    """
    tmpdir = tempfile.TemporaryDirectory()
    pial_asc = os.path.join(tmpdir.name, os.path.basename(pial_surf + ".asc"))
    subprocess.run(["mris_convert", pial_surf, pial_asc])

    with open(pial_asc, "r") as f:
        f.readline()
        nverts, ntriangs = [int(n) for n in f.readline().strip().split(" ")]

    vertices = np.genfromtxt(
        pial_asc, dtype=float, skip_header=2, skip_footer=ntriangs, usecols=(0, 1, 2)
    )
    triangles = np.genfromtxt(
        pial_asc, dtype=int, skip_header=2 + nverts, usecols=(0, 1, 2)
    )
    assert vertices.shape == (nverts, 3)
    assert triangles.shape == (ntriangs, 3)

    completed_process = subprocess.run(
        ["mris_info", pial_surf], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    mris_info = completed_process.stdout.decode("ascii")
    c_ras_list = extract_vector(mris_info, "c_(ras)")
    assert c_ras_list is not None
    vertices[:, 0:3] += np.array(c_ras_list)

    return vertices, triangles


def read_cortical_region_mapping(
    label_direc: os.PathLike, hemisphere: Hemisphere, fs_to_conn: RegionIndexMapping
) -> np.ndarray:
    """
    Read the cortical region mapping file.

    :param label_direc: Where the annotation label directory is
    :param hemisphere: (Hemisphere) enumerator
    :param fs_to_conn: (RegionIndexMapping)
    :return:
    """
    filename = os.path.join(label_direc, hemisphere.value + ".aparc.annot")
    region_mapping, _, _ = nibabel.freesurfer.io.read_annot(filename)
    region_mapping = region_mapping - 1
    region_mapping[region_mapping == -2] = 0  # Unknown regions in hemispheres

    # $FREESURFER_HOME/FreeSurferColorLUT.txt describes the shift
    if hemisphere == Hemisphere.lh:
        region_mapping += FS_LUT_LH_SHIFT
    else:
        region_mapping += FS_LUT_RH_SHIFT

    fs_to_conn_fun = np.vectorize(lambda n: fs_to_conn.source_to_target(n))
    region_mapping = fs_to_conn_fun(region_mapping)

    return region_mapping
