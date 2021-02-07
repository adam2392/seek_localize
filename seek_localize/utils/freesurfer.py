import glob
import os
from pathlib import Path
from typing import Union, Dict, Any

import nibabel as nb
import numpy as np
import scipy.io
import scipy.spatial
from nptyping import NDArray

from seek_localize.utils.projection import project_electrodes_anydirection


def _read_vertex_labels(gyri_labels_dir: Union[str, Path], hem: str) -> Dict[int, str]:
    """Read anatomical labels of gyri vertices.

    Parameters
    ----------
    gyri_labels_dir : str | pathlib.Path
        Directory of the gyri labels. Usually ``<FreeSurferSubjectDir>/<subject>/gyri/``.
    hem : str
        Either ``'rh'``, or ``'lh'``.

    Returns
    -------
    vert_label : dict
        Dictionary of vertex indices (key) and anatomical labels (value).
    """
    # Use the surface label files to get which label goes with each surface vertex
    label_files = glob.glob(os.path.join(gyri_labels_dir, "%s.*.label" % (hem)))

    # construct dictionary of vertices and their labels
    vert_label = {}
    for label in label_files:
        label_name = label.split(".")[1]
        print("Loading label %s" % label_name)
        fid = open(label, "r")
        d = np.genfromtxt(fid, delimiter=" ", skip_header=2)
        vertnum, x, y, z, junk = d[~np.isnan(d)].reshape((-1, 5)).T
        for v in vertnum:
            vert_label[int(v)] = label_name.strip()
        fid.close()
    return vert_label


def _read_cortex_vertices(
    mesh_dir: Path,
    hem: str,
    on_error: str = "raise",
    surf_type: str = "pial",
    return_tri: bool = False,
) -> NDArray[Any, 3, Any]:
    """Read vertices of the pial surface of the FreeSurfer generated cortex.

    Parameters
    ----------
    mesh_dir : str | pathlib.Path
        The directory corresponding to the mesh directory. Must be
        ``<FreeSurferSubjectDir>/<subject>/Meshes/``.
    hem : str
        Either ``'rh'``, or ``'lh'``.
    on_error : str
        One of 'raise', or 'ignore'. If ``raise``, then if the ``<hem>_pial_trivert.mat``
        file is not found, then a ValuError will be raised. If ``ignore``, then
        :func:`convert_fsmesh2mlab` will be run.
    surf_type : str
        One of 'pial', 'dural', 'OFC'.
    Returns
    -------
    cortex_tris : np.ndarray
        The N x 3 array of vertex indices per triangle.
    cortex_verts : np.ndarray
        The N x 3 array of vertices and their 3D locations.
    """
    if on_error not in ["raise", "ignore"]:
        raise ValueError(
            f'"on_error" can only be "raise", or "ignore", not ' f"{on_error}."
        )

    # load in hemispheric pial triangular-vertices mat files
    trivert_file = os.path.join(mesh_dir, f"{hem}_{surf_type}_trivert.mat")
    if not os.path.exists(trivert_file):
        if on_error == "raise":
            raise FileNotFoundError(
                "Trivert file .mat file was not created yet. "
                "Please call `convert_fsmesh2mlab` function first."
            )
        elif on_error == "ignore":
            subj = mesh_dir.parent.name
            fs_subj_dir = mesh_dir.parent.parent
            convert_fsmesh2mlab(fs_subj_dir, subj, mesh_name="pial")

    # load the mat struct
    cortex_verts = scipy.io.loadmat(trivert_file)["vert"]
    cortex_tris = scipy.io.loadmat(trivert_file)["tri"]
    if return_tri:
        return cortex_tris, cortex_verts
    return cortex_verts


def convert_fsmesh2mlab(
    subj_dir, subj: str, mesh_name: str = "pial"
) -> Union[Dict[str, str], str]:
    """Create surface mesh triangle and vertex .mat files.

    If no argument for mesh_name is given, lh.pial and rh.pial
    are converted into lh_pial_trivert.mat and rh_pial_trivert.mat
    in the Meshes directory (for use in python) and *_lh_pial.mat
    and *_rh_pial.mat for use in MATLAB.

    Parameters
    ----------
    subj_dir : str | pathlib.Path
        The ``FreeSurferSubjectDir``.
    subj : str
        The subject identifier inside ``subj_dir``.
    mesh_name : str
        One of {'pial', 'white', 'inflated', 'dural'}.

    Returns
    -------
    out_file : dict | str
        If mesh_name == ``'pial'``, then out_file will be a dictionary
        of the ``'rh'``, and ``'lh'`` hemisphere pial files. Else
        it will be the output file.
    """
    surf_dir = os.path.join(subj_dir, subj, "surf")
    mesh_dir = os.path.join(subj_dir, subj, "Meshes")
    hems = ["lh", "rh"]

    if not os.path.isdir(mesh_dir):
        print("Making Meshes Directory")
        # Make the Meshes directory in subj_dir if it does not yet exist
        os.mkdir(mesh_dir)

    # Loop through hemispheres for this mesh, create one .mat file for each
    for hem in hems:
        print("Making %s mesh" % (hem))
        mesh_surf = os.path.join(surf_dir, f"{hem}.{mesh_name}")

        # read vertices and triangles array data structures from
        # surface mesh files
        vert, tri = nb.freesurfer.read_geometry(mesh_surf)

        # save these to matlab-compatible files
        out_file = os.path.join(mesh_dir, f"{hem}_{mesh_name}_trivert.mat")
        scipy.io.savemat(out_file, {"tri": tri, "vert": vert})

        # save matlab compatible file (tri is indexed at 1)
        out_file_struct = os.path.join(
            mesh_dir, "%s_%s_%s.mat" % (subj, hem, mesh_name)
        )
        cortex = {"tri": tri + 1, "vert": vert}
        scipy.io.savemat(out_file_struct, {"cortex": cortex})

    out_file_dict = dict()
    if mesh_name in ["pial", "dural"]:
        out_file_dict["lh"] = os.path.join(mesh_dir, f"lh_{mesh_name}_trivert.mat")
        out_file_dict["rh"] = os.path.join(mesh_dir, f"rh_{mesh_name}_trivert.mat")
        return out_file_dict
    else:
        return out_file


def project_electrodes(
    elec_coords: NDArray[(Any, 3), float],
    fs_subj_dir: Union[str, Path],
    hem: str,
    surf_type: str = "dural",
    use_mean_normal: bool = False,
    convex_hull: bool = True,
    verbose: bool = True,
):
    fs_subj_dir = Path(fs_subj_dir)
    subj = fs_subj_dir.name

    dural_mesh = fs_subj_dir / "Meshes" / f"{subj}_{hem}_dural.mat"
    # if surf_type == 'dural' and not os.path.isfile(dural_mesh):
    #     print('Creating dural surface mesh, using %d smoothing iterations' % (num_iter))
    # make_dural_surf(num_iter=num_iter, dilate=dilate)

    # compute projection direction
    if use_mean_normal:
        # get corners of the grid electrode

        # get four normal vectors

        # normalize them and get the mean of the normal vectors

        print("not implemented yet...")
        direction = hem
    else:
        direction = hem

    # load cortex's vertices and triangles
    tri, vert = _read_cortex_vertices(
        mesh_dir=fs_subj_dir / "Meshes", hem=hem, surf_type=surf_type, return_tri=True
    )

    # create the convex hull
    if convex_hull:
        # create delauney tessalation in 3 dimensions
        tri_tess = scipy.spatial.Delaunay(vert)
        tri = tri_tess.convex_hull

    # project the electrodes
    elecs_proj = project_electrodes_anydirection(tri, vert, elec_coords, direction)
    return elecs_proj


def prepare_mesh_cortexhull(
    fs_subj_dir: Union[str, Path],
    smooth_steps: int = 30,
    expansion_mm: Any = 0,
    resolution: float = 1,
    fix_shrinkage: bool = False,
    verbose: bool = True,
):
    """Prepare cortex hull for FreeSurfer subject pial surface mesh.

    See [1] for similar implementation in Fieldtrip.

    Uses FreeSurfer commands: ``mris_fill`` and ``mris_smooth``
    along with other algorithms like Marching Cubes.

    From the ``surf/<hem>.pial`` files, generates the following new
    files:

    1. Pial filled image: ``surf / <hem>.pial.filled.mgz``
    2. Dural smoothed surface: ``surf / <hem>.dural``
    3. Outer pial surface: ``surf / <hem>.pial.outer``

    Parameters
    ----------
    fs_subj_dir : str | pathlib.Path
        FreeSurfer subject directory.
    smooth_steps : int
        Number of standard smoothing iterations (default: 0)
    expansion_mm : float | 'auto'
        Amount in mm with which the hull is re-expanded, applies automatically
        when ``fix_shrinkage = True`` (default = 1).
    resolution : float
        Resolution of the volume delimited by headshape being floodfilled.
        to pass into the ``mris_fill`` (default=1),
    fix_shrinkage : bool
        Whether to try to fix shrinkage (default=False).
    verbose : bool
        Verbosity.

    References
    ----------
    .. [1] See FieldTrip toolbox ``ft_prepare_mesh.m`` file.
    """
    fs_subj_dir = Path(fs_subj_dir)

    # Create mask of pial surface
    hems = ["lh", "rh"]
    for hem in hems:
        if verbose:
            print(f"Creating mask of {hem} pial surface")

        # pial surface file for this hemisphere
        pial_surf = fs_subj_dir / "surf" / f"{hem}.pial"

        # new surface image with filled pial surface
        pial_fill_image = fs_subj_dir / "surf" / f"{hem}.pial.filled.mgz"

        # dural smoothed surface file per hemisphere
        dura_surf = fs_subj_dir / "surf" / f"{hem}.dural"

        # outer pial surface file
        outfile = fs_subj_dir / "surf" / f"{hem}.pial.outer"

        # if filled pial image hasn't been generated, create it
        if not os.path.isfile(pial_fill_image):
            os.system(f"mris_fill -c -r {resolution} {pial_surf} {pial_fill_image}")

        if not os.path.isfile(pial_fill_image):  # noqa
            raise RuntimeError(
                f'FreeSurfer "mris_fill" did not run to '
                f"fill a pial surface image. The image should "
                f"have been created at {pial_fill_image}. Please "
                f"check your FreeSurfer installation."
            )

        # Create outer surface of this pial surface by closing the gaps
        if verbose:
            print(
                f"Creating outer surface for the filled "
                f"pial surface {pial_fill_image}"
            )
        if not os.path.exists(outfile):
            make_outer_surf(pial_surf, pial_fill_image, outfile, expansion_mm)

        # smooth the surface using FreeSurfer's mris_smooth function (not applied by default)
        os.system("mris_extract_main_component %s %s-main" % (outfile, outfile))
        os.system(f"mris_smooth -nw -n {smooth_steps} {outfile}-main {dura_surf}")

        # fix shrinkage if needed. Try to expand the surface by a dilation
        # factor
        if fix_shrinkage:
            # mean outside distance
            if expansion_mm == "auto":
                expansion_mm = 0.0
                raise NotImplementedError("not done yet...")
                # expansion_mm = np.mean(expansion[idx])

            if expansion_mm != 0:
                if verbose:
                    print(f"Dilating surface by {expansion_mm} mm")
                if expansion_mm > 0:
                    if verbose:
                        print("Multiplying dilate value by -1 to get outward dilation")
                    expansion_mm = -1 * expansion_mm

                # global surface expansion
                os.system(f"mris_expand {dura_surf} {expansion_mm} {dura_surf}")
            else:
                print("Failed to create %s, check inputs." % (pial_fill_image))

    # run conversion of fsmesh
    subj = fs_subj_dir.name
    subj_dir = fs_subj_dir.parent
    convert_fsmesh2mlab(subj_dir=subj_dir, subj=subj, mesh_name="dural")


def make_outer_surf(
    orig_pial: Union[str, Path],
    image: Union[str, Path],
    output_fpath: Union[str, Path],
    outer_surface_sphere: float = 15,
):
    """Create outer surface of a pial volume.

    Make outer surface based on a pial volume and radius,
    write to surface in outfile.

    Parameters
    ----------
    orig_pial : str | pathlib.Path
        Pial surface (e.g. lh.pial)
    image : str | pathlib.Path
        Filled lh or rh pial image (e.g. lh.pial.filled.mgz)
    output_fpath : str | pathlib.Path
        surface file to write data to
    outer_surface_sphere : float | None
        radius for smoothing in mm (default=15). diameter of the sphere used
        by make_outer_surface to close the sulci using morphological operations.
        Ignored currently. Corresponds to ``se=strel('sphere',se_diameter);``
        in Matlab. See [1].

    References
    ----------
    .. [1] See FieldTrip Toolbox ``make_outer_surface`` function inside
    ``prepare_mesh_cortexhull.m`` file.

    .. [2] https://github.com/aestrivex/ielu
    """
    from scipy.signal import convolve
    from scipy.ndimage.morphology import grey_closing, generate_binary_structure
    from mne import write_surface
    from mcubes import marching_cubes

    # radius information is currently ignored
    # it is a little tougher to deal with the morphology in python

    # load original pial surface to get the volume information
    pial_surf = nb.freesurfer.read_geometry(orig_pial, read_metadata=True)
    volume_info = pial_surf[2]

    # load filled pial image
    fill = nb.load(image)
    filld = fill.get_data()
    filld[filld == 1] = 255

    # apply a very soft Gaussian filter with sigma = 1mm to
    # facilitate the closing
    gaussian = np.ones((2, 2)) * 0.25

    # initialize image cube array
    image_f = np.zeros((256, 256, 256))

    # initialize a thresholded image
    image2 = np.zeros((256, 256, 256))

    # for each slice, convolve the Gaussian filter on the
    # filled image
    for slice in range(256):
        temp = filld[:, :, slice]
        image_f[:, :, slice] = convolve(temp, gaussian, "same")

    # thresholded image based on value of 25
    image2[np.where(image_f <= 25)] = 0
    image2[np.where(image_f > 25)] = 255

    strel15 = generate_binary_structure(3, 1)

    # run multi-dimensional grayscale closing of the image
    BW2 = grey_closing(image2, structure=strel15)
    thresh = np.max(BW2) / 2
    BW2[np.where(BW2 <= thresh)] = 0
    BW2[np.where(BW2 > thresh)] = 255

    # apply marching cubes algorithm to get
    # vertices and faces
    v, f = marching_cubes(BW2, 100)

    # in order to cope with the different orientation
    v2 = np.transpose(
        np.vstack(
            (
                128 - v[:, 0],
                v[:, 2] - 128,
                128 - v[:, 1],
            )
        )
    )

    write_surface(output_fpath, v2, f, volume_info=volume_info)
