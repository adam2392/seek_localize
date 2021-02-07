import argparse
import collections
import re
import warnings
from pathlib import Path
from pprint import pprint
from typing import List, Dict

import nibabel as nb
import numpy as np
from scipy import spatial
from scipy.stats import norm
from skimage import measure
from sklearn.base import BaseEstimator


def _get_entry_exit_contacts(electrodes: Electrodes):
    entry_exit_elec = collections.defaultdict(list)
    for elec in electrodes:
        if len(elec) < 6:
            warnings.warn(
                f"Channels on electrode {elec} contain less than 6 contacts - {len(elec)}. "
                "Were endpoints correctly labeled?"
            )

        # get entry/exit contacts
        entry_ch = elec.get_entry_ch()
        exit_ch = elec.get_exit_ch()
        entry_exit_elec[elec.name] = [entry_ch, exit_ch]

        # remove all but the entry/exit
        for ch in elec.contacts:
            if ch.name not in [entry_ch.name, exit_ch.name]:
                elec.remove_contact(ch.name)

    return entry_exit_elec


def _visualize_electrodes(img, clusters, radius, threshold, output_fpath):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # get limits based on the image shape in voxels
    xlim, ylim, zlim = img.shape
    axcodes = nb.aff2axcodes(img.affine)

    # rotate the axes and update
    angles = [0, 45, 90, 180, 270, 360]

    # create figure
    fig, axs = plt.subplots(3, 2, figsize=(30, 30), subplot_kw=dict(projection="3d"))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    mpl_colors_iter = prop_cycle.by_key()["color"]
    colors = []
    for i, color in enumerate(mpl_colors_iter):
        if i == len(clusters.keys()):
            break
        colors.append(color)

    colors = cm.rainbow(np.linspace(0, 1, len(clusters.keys())))
    print("Trying to plot electrode 3D distributions...")
    print(len(colors))
    print(len(clusters.keys()))

    # generate a rotated figure for each plot
    for j, angle in enumerate(angles):
        ax = axs.flat[j]
        # ax = Axes3D(fig)
        ax.view_init(azim=angle)
        # label_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.5']
        for i, electrode in enumerate(clusters):
            clusters_i = []
            clusters_j = []
            clusters_k = []
            for cluster_id in clusters[electrode]:
                for point in clusters[electrode][cluster_id]:
                    clusters_i.append(point[0])
                    clusters_j.append(point[1])
                    clusters_k.append(point[2])
            ax.scatter3D(
                clusters_i,
                clusters_j,
                clusters_k,
                color=colors[i],
                label=electrode,
                cmap="",
            )
        ax.set_xlim3d(0, xlim + 1)
        ax.set_ylim3d(0, ylim + 1)
        ax.set_zlim3d(0, zlim + 1)
        ax.set_title(
            f"Azim-Angled {angle} Post-Processing Contacts \n"
            "(Radius = %d voxels, Threshold = %.3f)" % (radius, threshold)
        )
        ax.set_xlabel(f"(x) {axcodes[0]}")
        ax.set_ylabel(f"(x) {axcodes[1]}")
        ax.set_zlabel(f"(x) {axcodes[2]}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.2, 0.8), shadow=True, ncol=2)

    fig.tight_layout()
    plt.savefig(output_fpath, box_inches="tight")
    plt.close(fig)


class CylinderGroupMixin:
    """Cylindrical boundary mixin functions."""

    def compute_cylindrical_clusters(
        self, clustered_voxels, entry_point_vox, exit_point_vox, radius
    ):
        """Group clusters of points into cylinders based on two points.

        Parameters
        ----------
        clustered_voxels : dict
            Dictionary of cluster IDs (key) and list of voxels (value).
        entry_point_vox :
        exit_point_vox :
        radius : float
            Radius of the cylinders to consider. Think of this parameter
            as an ``epsilon``-ball around the voxel cloud of points. It
            should be slightly larger then the radius of the actual contacts
            in consideration.

        Returns
        -------
        voxel_clusters_in_cylinder : dict
            Dictionary of cluster IDs (key) associated with a cylinder
             and list of voxels (value).
        """
        # each electrode corresponds now to a separate cylinder
        voxel_clusters_in_cylinder = {}

        # go through each electrode apply cylinder filter based on entry/exit point
        # remove all voxels in clusters that are not within cylinder
        for cluster_id, cluster_voxels in clustered_voxels.items():
            # Obtain all points within the electrode endpoints pt1, pt2
            contained_voxels = []
            for voxel in cluster_voxels:
                if self._is_point_in_cylinder(
                    entry_point_vox, exit_point_vox, radius, voxel
                ):
                    contained_voxels.append(voxel)
            if not contained_voxels:
                continue

            # store all voxel clusters that are within this cylinder
            voxel_clusters_in_cylinder[cluster_id] = np.array(cluster_voxels)
        return voxel_clusters_in_cylinder

    def _is_point_in_cylinder(self, pt1, pt2, radius: int, q):
        """Test whether a point q lies within a cylinder.

        With points pt1 and pt2 that define the axis of the cylinder and a
        specified radius r. Used formulas provided here [1].

        Parameters
        ----------
        pt1: ndarray
            first point to bound the cylinder. (N, 1)
        pt2: ndarray
            second point to bound the cylinder. (N, 1)
        radius: int
            the radius of the cylinder.
        q: ndarray
            the point to test whether it lies within the cylinder.

        Returns
        -------
            True if q lies in the cylinder of specified radius formed by pt1
            and pt2, False otherwise.

        References
        ----------
        .. [1] https://www.flipcode.com/archives/Fast_Point-In-Cylinder_Test.shtml
        """
        # convert to arrays
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        q = np.array(q)

        if any([x.shape != pt1.shape for x in [pt2, q]]):  # pragma: no cover
            print(pt1, pt2, q)
            raise RuntimeError(
                "All points that are checked in cylinder "
                "should have the same shape. "
                f"The shapes passed in were: {pt1.shape}, {pt2.shape}, {q.shape}"
            )

        vec = pt2 - pt1
        length_sq = np.linalg.norm(vec) ** 2
        radius_sq = radius ** 2
        testpt = q - pt1  # set pt1 as origin
        dot = np.dot(testpt, vec)

        # Check if point is within caps of the cylinder
        if dot >= 0 and dot <= length_sq:
            # Distance squared to the cylinder axis of the cylinder:
            dist_sq = np.dot(testpt, testpt) - (dot * dot / length_sq)
            if dist_sq <= radius_sq:
                return True

        return False


class PostProcessMixin:
    """Post processing mixin functions."""

    def _identify_merged_voxel_clusters(
        self,
        voxel_clusters: Dict[str, np.ndarray],
        lb_size: int = 50,
        ub_size: int = 200,
    ) -> List[int]:
        """
        Classify the abnormal clustered_voxels that are extremely large.

        Uses a lower and upper bound of number of voxels to determine if
        cluster is merged or not.

        Parameters
        ----------
        voxel_clusters: dict(str: ndarray)
            Dictionary of clustered_voxels sorted by the cylinder/electrode
            in which they fall. The keys of the dictionary are electrode
            labels, the values of the dictionary are the cluster points
            from the threshold-based clustering algorithm that fell
            into a cylinder.
        lb_size : int
            Lower bound on the threshold of the size of merged clusters.
        ub_size : int
            Upper bound on the threshold of the size of merged clusters.

        Returns
        -------
        merged_cluster_ids: List[int]
            List of cluster ids thought to be large due to lack of
            sufficient separation between two channels in image.
        """
        merged_cluster_ids = []

        for cluster_id, points in voxel_clusters.items():
            # Average size of normal cluster is around 20-25 points
            cluster_size = len(points)

            # Merged clustered_voxels are likely to be moderately large
            if lb_size <= cluster_size <= ub_size:
                merged_cluster_ids.append(cluster_id)

        return merged_cluster_ids

    def _identify_skull_voxel_clusters(
        self, voxel_clusters: Dict, skull_cluster_size: int = 200
    ):
        """
        Identify abnormal clustered_voxels that are extremely large and near skull.

        TODO: identify convex hull of the brain

        Parameters
        ----------
        voxel_clusters: dict(str: ndarray)
            Dictionary of clustered_voxels sorted by the cylinder/electrode
            in which they fall.

            The keys of the dictionary are channel names,
            the values of the dictionary are the cluster points
            from the threshold-based clustering algorithm that fell
            into a cylinder.

        Returns
        -------
        skull_cluster_ids: dict(str: List[int])
            Dictionary of cluster ids thought to be large due to close
            proximity to the skull.
        """
        skull_clusters = []

        for cluster_id, voxels in voxel_clusters.items():
            # Average size of normal cluster is around 20-25 points
            cluster_size = len(voxels)

            # Skull clustered_voxels are likely to be very large
            if cluster_size > skull_cluster_size:
                skull_clusters.append(cluster_id)

        return skull_clusters

    def _pare_cluster(self, points_in_cluster, qtile, centroid=None):
        """Pare a cluster down by a quantile around a centroid."""
        if centroid is None:
            # get the centroid of that cluster
            centroid = np.mean(points_in_cluster, keepdims=True)

        # compute spatial variance of the points inside cluster
        var = np.var(points_in_cluster, axis=0)

        # store the pared clsuters
        pared_cluster = []

        # Include points that have a z-score within specified quantile
        for pt in points_in_cluster:
            # Assuming the spatial distribution of points is
            # approximately Gaussian, the outermost channel will be
            # approximately the centroid of this cluster.
            diff = pt - centroid
            z = np.linalg.norm(np.divide(diff, np.sqrt(var)))
            if norm.cdf(z) <= qtile:
                pared_cluster.append(pt)
        return pared_cluster

    def _pare_clusters_on_electrode(self, voxel_clusters, skull_cluster_ids, qtile):
        """
        Pare down skull clustered_voxels.

        Only considering points close to the
        centroid of the oversized cluster.

        Parameters
        ----------
        voxel_clusters: dict(str: dict(str: ndarray))
            Dictionary of clustered_voxels sorted by the cylinder/electrode
            in which they fall. The keys of the dictionary are electrode
            labels, the values of the dictionary are the cluster points
            from the threshold-based clustering algorithm that fell
            into a cylinder.

        skull_cluster_ids: dict(str: List[int])
            Dictionary of cluster ids thought to be large due to close
            proximity to the skull.

        qtile: float
            The upper bound quantile distance that we will consider for
            being a part of the resulting pared clustered_voxels.

        Returns
        -------
        voxel_clusters: dict(str: dict(str: ndarray))
            Dictionary of skull clustered_voxels that have been resized.
        """
        # for elec in skull_cluster_ids:
        # Get the coordinate for the outermost labeled channel from user in
        # last_chan_coord = list(sparse_labeled_contacts.values())[-1]
        for cluster_id in skull_cluster_ids:
            # get the clustered points for this cluster ID
            points_in_cluster = voxel_clusters[cluster_id]

            # pare the cluster based on quantile near the centroid
            pared_cluster = self._pare_cluster(points_in_cluster, qtile, centroid=None)

            # Sanity check that we still have a non-empty list
            if pared_cluster != []:
                voxel_clusters[cluster_id] = np.array(pared_cluster)

        return voxel_clusters

    def _compute_convex_hull(self, masked_img: nb.Nifti2Image):
        img_arr = masked_img.get_fdata()
        voxel_pts = img_arr[img_arr > 0]
        cvxhull = spatial.ConvexHull(voxel_pts)


class SEEGLocalizer(BaseEstimator, CylinderGroupMixin, PostProcessMixin):
    """Localization algorithm for detecting channel centroids for sEEG electrodes.

    Requires at least two channel points on each electrode.

    Algorithm proceeds as follows:

    1. Apply a brainmask to the CT image (optional, but RECOMMENDED) to
    remove parts of the skull.
    2. Cluster bright voxel points on the CT image.
    3. Apply initialization double-points on each electrode
    4. Apply a cylindrical boundary to further group clusters into
    electrode groups defined by a cylinder around the initialization points.
    4. Recursively cluster until reached number of contacts.
    5. Number electrodes accordingly.

    Parameters
    ----------
    ct_img : nb.Nifti2Image
    radius : int
        Radius in CT voxels of the cylindrical boundary.
    threshold : float
        Threshold to apply to voxel values. Between 0 and 1. Turns any
        voxel values < threshold to 0.
    contact_spacing : float | None
        The spacing between contacts on the electrode in mm. Only applicable
        if known. E.g. 3.5 mm
    brainmask_img : nb.Nifti2Image | None
    """

    def __init__(
        self,
        ct_img,
        radius=4,
        threshold=0.630,
        contact_spacing=None,
        brainmask_img=None,
        verbose: bool = True,
    ):
        self.ct_img = ct_img
        self.brainmask_img = brainmask_img
        self.radius = radius
        self.threshold = threshold
        self.contact_spacing = contact_spacing
        self.verbose = verbose

    def _get_masked_space(self) -> nb.Nifti2Image:
        """
        Apply a binarized brain mask to an input CT image file.

        Returns
        -------
        masked_ct_img: NiBabel Image Object
            Masked CT NiBabel Image Object in the same voxel space
            as input.
        """
        if self.brainmask_img is None:
            return self.ct_img

        # obtain image arrays
        ct_img_arr = self.ct_img.get_fdata()
        mask_arr = self.brainmask_img.get_fdata()

        # binarize mask to 0 and 1
        mask_arr[mask_arr > 0] = 1

        # multiply element wise brain array and mask array
        masked_brain = np.multiply(ct_img_arr, mask_arr)

        # return nibabel image
        masked_ct_img = nb.Nifti2Image(masked_brain, self.ct_img.affine)
        return masked_ct_img

    def _compute_voxel_clusters(self, threshold):
        """Compute clusters of voxels on CT image.

        Apply a thresholding based algorithm and then run connected
        clustering using ``skimage``.

        This will then return clustered_voxels of voxels that belong
        to distinct contact groups.
        http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

        Parameters
        ----------
        threshold: float
            Brightness threshold to use for binarizing the input image.

        Returns
        -------
        clustered_voxels: dict
            Dictionary that maps cluster ID's (key) to list of voxels (value)
            belonging to that cluster.
        num_obj: int
            Number of clustered_voxels found in image for specified threshold.
        """
        if threshold < 0 or threshold > 1.0:  # pragma: no cover
            raise ValueError(
                "Threshold for clustering should be between 0 and 1, since "
                "threshold is based on the normalized voxel brightness (i.e. "
                f"all voxel values are normalized by 255. You passed in {threshold}."
            )
        brainmasked_CT_arr = self._get_masked_space().get_fdata()

        # initialize clustered_voxels as a dictionary by ID
        clusters = {}

        if self.verbose:
            print(f"Computing clusters using threshold: {threshold}")

        # Label voxels with ID corresponding to the cluster it belongs to
        cluster_labels, num_obj = measure.label(
            brainmasked_CT_arr / 255 > threshold,
            background=0,
            return_num=True,
            connectivity=2,
        )

        if self.verbose:
            print(f"Found {num_obj} clusters of voxels.")

        # Filter out all zero-valued voxels in cluster_labels (pixels)
        nonzeros = np.nonzero(cluster_labels)
        nonzero_voxs = np.array(list(zip(*nonzeros)))

        # Group voxels by their cluster ID
        for vox in nonzero_voxs:
            clusters.setdefault(cluster_labels[tuple(vox)], []).append(vox)
        assert len(clusters) == num_obj

        self.voxel_clusters = clusters
        return clusters, num_obj

    def fit(self, X, elecs, y=None):
        """Estimate full electrode contact locations in voxel space.

        Parameters
        ----------
        X : np.ndarray
            N x 6 array where N is the number of electrodes, and
            the first 3 coordinates are the entry points, and the
            last 3 coordinates are the exit points.
        y : np.ndarray | None
        """
        self._validate_data(X, y)

        n_elecs, _ = X.shape

        if self.verbose:
            print(f"Applying SEEK-localize algorithm for electrodes: {X}")

        # compute voxel clusters in the brain image using a threshold
        voxel_clusters, num_clusters = self._compute_voxel_clusters(
            threshold=self.threshold
        )

        # group clusters into semantic groups (electrode names)
        # based on a cylindrical boundary in the direction of the
        # 2 labeled contacts
        _cylindrical_clusters = {}
        for idx, elec_name in range(elecs):
            # get entry and exist points for each
            entry_point_vox, exit_point_vox = X[idx, :3], X[idx, -3:]

            # compute cylindrical boundary for each electrode
            voxel_clusters_in_cylinder = self.compute_cylindrical_clusters(
                clustered_voxels=voxel_clusters,
                entry_point_vox=entry_point_vox,
                exit_point_vox=exit_point_vox,
                radius=self.radius,
            )

            # each electrode now gets a list of voxel clusters
            # associated with it (from a cylindrical bounding)
            _cylindrical_clusters[elec_name] = voxel_clusters_in_cylinder

        # TODO: make work
        # recursively post-process these electrodes
        while 1:
            # assign sequential labels for the electrodes
            labeled_voxel_clusters = {}
            electrodes_with_problem = collections.defaultdict(list)
            for electrode in entry_exit_elec:
                elec_name = electrode.name
                entry_ch = electrode.get_entry_ch()
                exit_ch = electrode.get_exit_ch()
                this_elec_voxels = voxel_clusters[elec_name]

                _this_elec_voxels = brain.assign_sequential_labels(
                    this_elec_voxels, entry_ch.name, entry_ch.coord
                )

            # check each labels
            for ch_name in _this_elec_voxels.keys():
                _, ch_num = re.match("^([A-Za-z]+[']?)([0-9]+)$", ch_name).groups()
                if int(ch_num) not in elec_contact_nums[elec]:
                    electrodes_with_problem[elec_name].append(ch_num)

            # there were incorrectly grouped contact clusters
            if elec_name in electrodes_with_problem:
                print(
                    f"Electrode {elec_name} has incorrectly grouped clusters... "
                    f"Attempting to unfuse them."
                )
                print(this_elec_voxels.keys())

                # check for merged clusters at entry and exit for this electrode
                merged_cluster_ids = self._identify_merged_voxel_clusters(
                    this_elec_voxels
                )

                print(merged_cluster_ids)
                # if there are merged cluster ids, unfuse them
                this_elec_voxels = brain._unfuse_clusters_on_entry_and_exit(
                    this_elec_voxels, merged_cluster_ids, elec_contact_nums[elec]
                )

            # check for oversized clusters and pare them down
            oversized_clusters_ids = self._identify_skull_voxel_clusters(
                this_elec_voxels
            )

            print("Found oversized clusters: ", oversized_clusters_ids)

            # pare them down and resize
            this_elec_voxels = self._pare_clusters_on_electrode(
                this_elec_voxels, oversized_clusters_ids, qtile=0.5
            )

            # fill in gaps between centroids
            this_elec_voxels = brain.fill_clusters_with_spacing(
                this_elec_voxels,
                entry_ch,
                elec_contact_nums[elec_name],
                contact_spacing_mm=contact_spacing_mm,
            )

        self.estimated_elec_coords_ = this_elec_voxels
        return self


def _mainv2(
    ctimgfile,
    brainmaskfile,
    elecinitfile,
    brainmasked_ct_fpath=None,
    outputfig_fpath=None,
):
    # # Hyperparameters for electrode clustering algorithm
    # radius = 4  # radius (in CT voxels) of cylindrical boundary
    # threshold = 0.630  # Between 0 and 1. Zeroes voxels with value < threshold
    # contact_spacing_mm = 3.5  # distance between two adjacent contacts
    #
    # # load in nibabel CT and brainmask images
    # ct_img = nb.load(ctimgfile)
    # brainmask_img = nb.load(brainmaskfile)
    #
    # # initialize clustered brain image
    # brain = ClusteredBrainImage(ct_img, brainmask_img)
    # brain.save_masked_img(brainmasked_ct_fpath)  # save brain-masked CT file
    #
    # # load in the channel coordinates in xyz as dictionary
    # ch_coords_mm = load_elecs_data(elecinitfile)
    #
    # # convert into electrodes
    # ch_names = list(ch_coords_mm.keys())
    # ch_coords = list(ch_coords_mm.values())
    # electrodes = Electrodes(ch_names, ch_coords, coord_type="mm")
    #
    # # get the entry/exit electrodes
    # entry_exit_elec = get_entry_exit_contacts(electrodes)
    # # determine the contact numbering per electrode
    # elec_contact_nums = {}
    # for elec, (entry_ch, exit_ch) in entry_exit_elec.items():
    #     elec_contact_nums[elec] = _contact_numbers_on_electrode(
    #         entry_ch.name, exit_ch.name
    #     )
    #
    # # get sparse electrodes in voxel space
    # ch_names = []
    # ch_coords = []
    # # transform coordinates -> voxel space
    # for elec_name, contacts in entry_exit_elec.items():
    #     for contact in contacts:
    #         contact.transform_coords(brain.get_masked_img(), coord_type="vox")
    #         ch_names.append(contact.name)
    #         ch_coords.append(contact.coord)
    #         assert contact.coord_type == "vox"
    # entry_exit_elec = Electrodes(ch_names, ch_coords, coord_type="vox")

    # print("Applying SEEK algo... for electrodes: ", entry_exit_elec)
    # print("Contact numbering for each electrode: ", elec_contact_nums.keys())
    # # compute voxel clusters in the brain image using a threshold
    # voxel_clusters, num_clusters = brain.compute_clusters_with_threshold(
    #     threshold=threshold
    # )
    #
    # # feed in entry/exit voxel points per electrode and apply a cylinder filter
    # _cylindrical_clusters = {}
    # for electrode in entry_exit_elec:
    #     elec_name = electrode.name
    #     entry_point_vox = electrode.get_entry_ch().coord
    #     exit_point_vox = electrode.get_exit_ch().coord
    #     voxel_clusters_in_cylinder = brain.compute_cylindrical_clusters(
    #         voxel_clusters, entry_point_vox, exit_point_vox, radius=radius
    #     )
    #     _cylindrical_clusters[elec_name] = voxel_clusters_in_cylinder
    # voxel_clusters = _cylindrical_clusters
    # print("Cylindrical bounded electrode clustered_voxels: ", voxel_clusters.keys())

    # preliminarily label electrode voxel clusters
    labeled_voxel_clusters = {}
    electrodes_with_problem = collections.defaultdict(list)
    for electrode in entry_exit_elec:
        elec_name = electrode.name
        entry_ch = electrode.get_entry_ch()
        exit_ch = electrode.get_exit_ch()
        this_elec_voxels = voxel_clusters[elec_name]

        _this_elec_voxels = brain.assign_sequential_labels(
            this_elec_voxels, entry_ch.name, entry_ch.coord
        )

        # check each labels
        for ch_name in _this_elec_voxels.keys():
            _, ch_num = re.match("^([A-Za-z]+[']?)([0-9]+)$", ch_name).groups()
            if int(ch_num) not in elec_contact_nums[elec]:
                electrodes_with_problem[elec_name].append(ch_num)

        # there were incorrectly grouped contact clusters
        if elec_name in electrodes_with_problem:
            print(
                f"Electrode {elec_name} has incorrectly grouped clusters... "
                f"Attempting to unfuse them."
            )
            print(this_elec_voxels.keys())

            # check for merged clusters at entry and exit for this electrode
            merged_cluster_ids = brain._identify_merged_voxel_clusters(this_elec_voxels)

            print(merged_cluster_ids)
            # if there are merged cluster ids, unfuse them
            this_elec_voxels = brain._unfuse_clusters_on_entry_and_exit(
                this_elec_voxels, merged_cluster_ids, elec_contact_nums[elec]
            )

        # check for oversized clusters and pare them down
        oversized_clusters_ids = brain._identify_skull_voxel_clusters(this_elec_voxels)

        print("Found oversized clusters: ", oversized_clusters_ids)

        # pare them down and resize
        this_elec_voxels = brain._pare_clusters_on_electrode(
            this_elec_voxels, oversized_clusters_ids, qtile=0.5
        )

        # fill in gaps between centroids
        this_elec_voxels = brain.fill_clusters_with_spacing(
            this_elec_voxels,
            entry_ch,
            elec_contact_nums[elec_name],
            contact_spacing_mm=contact_spacing_mm,
        )

        if entry_ch.coord_type == "vox":
            entry_ch.transform_coords(brain.get_masked_img(), coord_type="mm")
        if exit_ch.coord_type == "vox":
            exit_ch.transform_coords(brain.get_masked_img(), coord_type="mm")
        this_elec_xyz = collections.defaultdict(list)
        for _cluster_id, voxels in this_elec_voxels.items():
            for coord in voxels:
                this_elec_xyz[_cluster_id].append(
                    brain.map_coordinates(coord, coord_type="mm")
                )
        # # compute the average / std contact-to-contact spacing
        # import numpy as np
        # dists = []
        # for cluster_id, voxels in this_elec_xyz.items():
        #     curr_centroid = np.mean(voxels, axis=0)
        #     if dists == []:
        #         prev_centroid = np.mean(voxels, axis=0)
        #         dists.append(0)
        #         continue
        #     dists.append(np.linalg.norm(curr_centroid - prev_centroid))
        #     prev_centroid = curr_centroid
        # print("Distribution of contact to contact spacing: ", np.mean(dists), np.std(dists))
        # this_elec_xyz = brain.correct_labeled_clusters(this_elec_xyz,
        #                                                   entry_ch,
        #                                                   exit_ch,
        #                                                   contact_spacing_mm=contact_spacing_mm)
        #

        # apply brute force correction
        # this_elec_xyz = brain.bruteforce_correctionv2(
        #     this_elec_xyz,
        #     entry_ch,
        #     exit_ch,
        #     contact_spacing_mm=contact_spacing_mm, num_contacts=len(elec_contact_nums[electrode.name])
        # )
        # _this_elec_voxels = collections.defaultdict(list)
        # for _cluster_id, coords in this_elec_xyz.items():
        #     for coord in coords:
        #         _this_elec_voxels[_cluster_id].append(brain.map_coordinates(coord, coord_type='vox'))
        # this_elec_voxels = _this_elec_voxels

        if entry_ch.coord_type == "mm":
            entry_ch.transform_coords(brain.get_masked_img(), coord_type="vox")
        if exit_ch.coord_type == "mm":
            exit_ch.transform_coords(brain.get_masked_img(), coord_type="vox")
        # assign sequential labels
        this_elec_voxels = brain.assign_sequential_labels(
            this_elec_voxels,
            entry_ch.name,
            entry_ch.coord,
        )

        # reset electrode clusters to that specific electrode
        labeled_voxel_clusters[elec_name] = this_elec_voxels

    # label the centroids
    labeled_voxel_centroids, labeled_xyz_centroids = convert_voxel_clusters_to_centroid(
        brain, entry_exit_elec, labeled_voxel_clusters
    )

    # visualize the electrode
    visualize_electrodes(
        brain.get_masked_img(),
        labeled_voxel_clusters,
        radius,
        threshold,
        outputfig_fpath,
    )
    return labeled_voxel_centroids, labeled_xyz_centroids


def _convert_voxel_clusters_to_centroid(brain, entry_exit_elec, labeled_voxel_clusters):
    # Compute centroids for each cluster
    labeled_voxel_centroids = brain.cluster_2_centroids(labeled_voxel_clusters)

    # Convert final voxels to xyz coordinates
    labeled_xyz_centroids = brain.vox_2_xyz(
        labeled_voxel_centroids, brain.get_masked_img().affine
    )

    # keep the end points, since they're already labeled
    if entry_exit_elec.coord_type == "vox":
        entry_exit_elec.transform_coords(brain.get_masked_img(), coord_type="mm")

    for electrode in entry_exit_elec:
        for contact in electrode.contacts:
            labeled_xyz_centroids[electrode.name][contact.name] = contact.coord

    return labeled_voxel_centroids, labeled_xyz_centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ct_nifti_img", help="The CT image volume in its original space."
    )
    parser.add_argument(
        "brainmask_native_file", help="Brain mask mapped to the CT image space."
    )
    parser.add_argument(
        "electrode_initialization_file",
        help="The electrode file with contacts localized to 2 points.",
    )
    parser.add_argument(
        "clustered_points_file",
        help="The output datafile with all the electrode points clustered.",
    )
    parser.add_argument("clustered_voxels_file", help="the voxels output datafile")
    parser.add_argument("binarized_ct_volume", help="The binarized CT volume.")
    parser.add_argument("fs_patient_dir", help="The freesurfer output diretroy.")
    parser.add_argument("outputfig_fpath")
    args = parser.parse_args()

    # Extract arguments from parser
    ct_nifti_img = args.ct_nifti_img
    brainmask_native_file = args.brainmask_native_file
    electrode_initialization_file = args.electrode_initialization_file
    clustered_points_file = args.clustered_points_file
    clustered_voxels_file = args.clustered_voxels_file
    binarized_ct_file = args.binarized_ct_volume
    fs_patient_dir = args.fs_patient_dir
    outputfig_fpath = args.outputfig_fpath

    # Create electrodes directory if not exist
    elecs_dir = Path(Path(fs_patient_dir) / "elecs")
    elecs_dir.mkdir(exist_ok=True)

    # Compute the final centroid voxels, centroid xyzs and the binarized CT image volume
    final_centroids_voxels, final_centroids_xyz = mainv2(
        ct_nifti_img,
        brainmask_native_file,
        electrode_initialization_file,
        binarized_ct_file,
        outputfig_fpath,
    )

    # Save output files
    print(f"Saving clustered xyz coords to: {clustered_points_file}.")
    print(f"Saving clustered voxels to: {clustered_voxels_file}.")
    pprint(final_centroids_xyz)

    # save data into bids sidecar-tsv files
    save_organized_elecdict_astsv(
        final_centroids_xyz, clustered_points_file, img_fname=ct_nifti_img
    )
    # save_organized_elecdict_astsv(final_centroids_voxels, clustered_voxels_file, img_fname=ct_nifti_img)

    # Save centroids as .mat file with attributes eleclabels
    save_organized_elecdict_asmat(
        final_centroids_xyz, clustered_points_file.replace(".tsv", ".mat")
    )
    save_organized_elecdict_asmat(
        final_centroids_voxels, clustered_voxels_file.replace(".tsv", ".mat")
    )
