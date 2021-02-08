from typing import Any, Tuple, Optional

import numpy as np
from nptyping import NDArray, Int


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
