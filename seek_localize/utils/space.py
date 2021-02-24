from typing import Any, Tuple

import numpy as np
from nptyping import NDArray

"""Copied from visbrain (http://visbrain.org/generated/visbrain.utils.mni2tal.html)."""


def _spm_matrix(p):
    """Matrix transformation.
    Parameters
    ----------
    p : array_like
        Vector of floats for defining each tranformation. p must be a vector of
        length 9.
    Returns
    -------
    Pr : array_like
        The tranformed array.
    """
    q = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    p.extend(q[len(p):12])

    # Translation t :
    t = np.array([[1, 0, 0, p[0]],
                  [0, 1, 0, p[1]],
                  [0, 0, 1, p[2]],
                  [0, 0, 0, 1]])
    # Rotation 1 :
    r1 = np.array([[1, 0, 0, 0],
                   [0, np.cos(p[3]), np.sin(p[3]), 0],
                   [0, -np.sin(p[3]), np.cos(p[3]), 0],
                   [0, 0, 0, 1]])
    # Rotation 2 :
    r2 = np.array([[np.cos(p[4]), 0, np.sin(p[4]), 0],
                   [0, 1, 0, 0],
                   [-np.sin(p[4]), 0, np.cos(p[4]), 0],
                   [0, 0, 0, 1]])
    # Rotation 3 :
    r3 = np.array([[np.cos(p[5]), np.sin(p[5]), 0, 0],
                   [-np.sin(p[5]), np.cos(p[5]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    # Translation z :
    z = np.array([[p[6], 0, 0, 0],
                  [0, p[7], 0, 0],
                  [0, 0, p[8], 0],
                  [0, 0, 0, 1]])
    # Translation s :
    s = np.array([[1, p[9], p[10], 0],
                  [0, 1, p[11], 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return np.linalg.multi_dot([t, r1, r2, r3, z, s])


def tal2mni(xyz):
    """Transform Talairach coordinates into MNI.
    Parameters
    ----------
    xyz : array_like
        Array of Talairach coordinates of shape (n_sources, 3)
    Returns
    -------
    xyz_r : array_like
        Array of MNI coordinates of shape (n_sources, 3)
    """
    # Check xyz to be (n_sources, 3) :
    if (xyz.ndim != 2) or (xyz.shape[1] != 3):
        raise ValueError("The shape of xyz must be (N, 3).")
    n_sources = xyz.shape[0]

    # Transformation matrices, different zooms above/below AC :
    rotn = np.linalg.inv(_spm_matrix([0., 0., 0., .05]))
    upz = np.linalg.inv(_spm_matrix([0., 0., 0., 0., 0., 0., .99, .97, .92]))
    downz = np.linalg.inv(_spm_matrix([0., 0., 0., 0., 0., 0., .99, .97, .84]))

    # Apply rotation and translation :
    xyz = np.dot(rotn, np.c_[xyz, np.ones((n_sources,))].T)
    tmp = np.array(xyz)[2, :] < 0.
    xyz[:, tmp] = np.dot(downz, xyz[:, tmp])
    xyz[:, ~tmp] = np.dot(upz, xyz[:, ~tmp])
    return np.array(xyz[0:3, :].T)


def mni2tal(xyz):
    """Transform MNI coordinates into Talairach.
    Parameters
    ----------
    xyz : array_like
        Array of MNI coordinates of shape (n_sources, 3)
    Returns
    -------
    xyz_r : array_like
        Array of Talairach coordinates of shape (n_sources, 3)
    """
    # Check xyz to be (n_sources, 3) :
    if (xyz.ndim != 2) or (xyz.shape[1] != 3):
        raise ValueError("The shape of xyz must be (N, 3).")
    n_sources = xyz.shape[0]

    # Transformation matrices, different zooms above/below AC :
    up_t = _spm_matrix([0., 0., 0., .05, 0., 0., .99, .97, .92])
    down_t = _spm_matrix([0., 0., 0., .05, 0., 0., .99, .97, .84])
    xyz = np.c_[xyz, np.ones((n_sources,))].T

    tmp = np.array(xyz)[2, :] < 0.
    xyz[:, tmp] = np.dot(down_t, xyz[:, tmp])
    xyz[:, ~tmp] = np.dot(up_t, xyz[:, ~tmp])
    return np.array(xyz[0:3, :].T)


"""API to compute in the 3D brain space."""


def nearest_electrode_vert(
        cortex_verts: np.ndarray, elec_coords: np.ndarray, return_dist: bool = False
) -> Tuple[NDArray[(Any,), int], NDArray[(Any, 3), int], NDArray[(Any, Any), float]]:  # type: ignore
    """Find vertex on a mesh that is closest to the given electrode coordinates.

    A mesh consists of a list of vertices (e.g. 3D coordinates) that are
    connected by edges (list of tuple of vertex indices). For a given
    list of 3D electrode coordinates, this function will find the
    vertex indices of the closes vertices, and the corresponding
    3D coordinates of those closest vertices.

    Parameters
    ----------
    cortex_verts : array-like
        [n_vertices x 3] matrix of vertices on the cortical surface mesh
    elec_coords : array-like
        [n_chans x 3] matrix of 3D electrode coordinates
    return_dist : bool
        Whether to return the corresponding distance matrix [n_chans x n_vertices]
        ordered in terms of the nearest vertices.
    Returns
    -------
    vert_inds : array-like
        [n_chans,] Array of vertex indices that are closest to each of the
        electrode.
    nearest_verts : array-like
        Coordinates for the nearest cortical vertices
    """
    n_chans = elec_coords.shape[0]
    n_vertices = cortex_verts.shape[0]

    # construct a distance matrix n_chans X n_vertices
    dist_matrix = np.zeros((n_chans, n_vertices))

    # Find the distance between each electrode and all possible vertices
    # on the surface mesh
    for idx in np.arange(n_chans):
        dist_matrix[idx, :] = np.sqrt(
            (elec_coords[idx, 0] - cortex_verts[:, 0]) ** 2
            + (elec_coords[idx, 1] - cortex_verts[:, 1]) ** 2
            + (elec_coords[idx, 2] - cortex_verts[:, 2]) ** 2
        )

    # Find the index of the vertex nearest to each electrode
    vert_inds = np.argmin(dist_matrix, axis=1)
    nearest_verts = cortex_verts[vert_inds, :]
    vert_dists = dist_matrix[:, vert_inds]

    if return_dist:
        return vert_inds, nearest_verts, vert_dists
    else:
        return vert_inds, nearest_verts  # type: ignore


if __name__ == '__main__':
    from pathlib import Path
    from mne_bids import BIDSPath
    from seek_localize import write_dig_bids
    import pandas as pd

    datadir = '~/Downloads/nihcoords'
    fname = 'NIH040.csv'

    subject = 'pt1'
    session = 'presurgery'
    datatype = 'ieeg'
    root = '/Users/adam2392/Dropbox/epilepsy_bids'
    # root = "/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_bids/"

    fpath = Path(datadir) / fname

    ch_tal_df = pd.read_csv(fpath, index_col=None)

    ch_names = ch_tal_df['chanName'].values
    xyz_coords = np.vstack([ch_tal_df[col] for col in ['x', 'y', 'z']]).T

    mni_coords = tal2mni(xyz_coords)
    ch_mni_df = ch_tal_df.copy()
    ch_mni_df['x'] = mni_coords[:, 0]
    ch_mni_df['y'] = mni_coords[:, 1]
    ch_mni_df['z'] = mni_coords[:, 2]

    bids_path = BIDSPath(subject=subject, session=session,
                         suffix='electrodes', extension='.tsv',
                         space='fsaverage',
                         root=root,
                         datatype=datatype)
    ch_mni_df.to_csv()

    write_dig_bids(bids_path.basename, root=root,
                   ch_names=ch_names, ch_coords=mni_coords,
                   unit='mm', coord_system='fsaverage',
                   overwrite=True)
    print(mni_coords)