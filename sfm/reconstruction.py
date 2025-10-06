import numpy as np
import cv2
from .triangulation import triangulate_points, filter_by_depth, filter_by_reprojection, filter_by_cheirality, triangulate_points_world_coords, filter_by_cheirality_world_coords
from .geometry import estimate_pose
import random
from scipy.spatial import cKDTree

class Camera:
    """Camera class storing intrinsics and extrinsics for SfM."""
    def __init__(self, K, R=None, t=None, kp=None):
        """
        Args:
            K (np.ndarray): 3x3 intrinsic camera matrix
            R (np.ndarray): 3x3 rotation matrix (world -> camera). Defaults to identity.
            t (np.ndarray): 3x1 translation vector (world -> camera). Defaults to zero vector.
            kp (list/np.ndarray): Keypoints in the image: 
        """
        self.K = K
        self.R = np.eye(3) if R is None else R
        self.t = np.zeros(3) if t is None else t
        self.kp = [] if kp is None else kp
        # Optional: map from keypoint index -> Point3D object (or None if unassigned), 
        # easier to find and merge observations between cameras
        self.observations = {}

    def projection_matrix(self):
        """
        Returns:
            np.ndarray: 3x4 projection matrix P = K[R|t] used for triangulation and reprojection
        """
        return self.K @ np.hstack((self.R, self.t.reshape(3,1)))

class Point3D:
    """3D point in space with references to observations in images."""
    _id_counter = 0
    def __init__(self, coord):
        """
        Args:
            coord (iterable): Initial 3D coordinates of the point
        """
        self.coord = np.array(coord, dtype=float)
        # Keep track of which cameras observe this point: {image_index: keypoint_index}
        self.observations = {}
        # Assigning ID for saving, filtering, etc
        self.id = Point3D._id_counter
        Point3D._id_counter += 1
        
class Reconstruction:
    """Incremental Structure-from-Motion (SfM) manager."""
    def __init__(self):
        # Dictionary of Camera objects representing reconstructed cameras #F
        self.cameras = {}
        # List of Point3D objects representing reconstructed 3D points
        #self.points3D = []
        self.points3D = {}  # id -> Point3D
        # Optional: store feature matches between image pairs
        # self.matches = {}

    def add_camera(self, camera, cam_idx):
        """Add a new camera to the reconstruction."""
        self.cameras[cam_idx] = camera
        print(f"Added camera: {cam_idx}")

    def add_point(self, point3D):
        """Add a new 3D point to the reconstruction."""
        self.points3D[point3D.id] = point3D

    def triangulate_new_points(self, cam1_idx, cam2_idx, kp1, kp2, matches, N=None, randomize=True, min_depth=0.1, max_depth=10.0):
        """
        Triangulate new 3D points from two views, and optionally only keep up to N of them.

        Args:
            cam1_idx (int): Index of first camera
            cam2_idx (int): Index of second camera
            kp1, kp2: Keypoints from each image
            matches (list[cv2.DMatch]): Matched keypoints between images
            N (int, optional): Max number of points to keep. If None, keep all.
            randomize (bool): If True, randomly subsample points. If False, keep first N.
        """
        cam1 = self.cameras[cam1_idx]
        cam2 = self.cameras[cam2_idx]

        # Triangulate + filters
        pts3D = triangulate_points_world_coords(kp1, kp2, matches, cam1, cam2)
        print("Before filtering:", len(pts3D))

        #Filter points in front of both cameras
        pts3D, matches_inlier = filter_by_cheirality_world_coords(pts3D, cam1, cam2, matches)
        print("After cheirality:", len(pts3D))

        # Filter by reasonable depth
        pts3D, matches_inlier = filter_by_depth(pts3D, min_depth=min_depth, max_depth=max_depth, matches=matches_inlier)
        print("After depth filter:", len(pts3D))

        # Filter by reprojection (Optional, removes many points if camera poses are still not accurate or not great matches)

        # pts3D, matches_inlier = filter_by_reprojection(pts3D, kp1, kp2, matches_inlier, cam2.R, cam2.t, cam1.K, thresh=15.0)
        # print("After reprojection filter:", len(pts3D))

        # === Subsample to N points if Coarse ===
        if N is not None and len(pts3D) > N:
            idxs = list(range(len(pts3D)))
            if randomize:
                idxs = random.sample(idxs, N)
            else:
                idxs = idxs[:N]

            pts3D = [pts3D[i] for i in idxs]
            matches_inlier = [matches_inlier[i] for i in idxs]
            print(f"Subsampled to {len(pts3D)} points (max N={N})")

        # Add each 3D point to reconstruction and record observations, merging pt as well
        for i, p in enumerate(pts3D):
            m = matches_inlier[i]
            existing_pt = cam1.observations.get(m.queryIdx, None)

            if existing_pt is not None:
                # Check if the old camera keypoint already has a 3D point
                existing_pt.observations[cam2_idx] = m.trainIdx
                cam2.observations[m.trainIdx] = existing_pt
            else:
                # Create new 3D point
                pt3d = Point3D(p)
                pt3d.observations[cam1_idx] = m.queryIdx
                pt3d.observations[cam2_idx] = m.trainIdx
                cam1.observations[m.queryIdx] = pt3d
                cam2.observations[m.trainIdx] = pt3d
                self.add_point(pt3d)

        return
    
    def run_two_view_init(self, kp1, kp2, matches, K, cam1_idx, cam2_idx, N=500, min_depth=0.1, max_depth=10.0):
        """
        Initialize reconstruction from precomputed keypoints and matches.

        Args:
            kp1, kp2: Lists of cv2.KeyPoint from image 1 and 2
            matches: List of cv2.DMatch
            K: Camera intrinsic matrix
        Returns:
            R, t, mask_pose: Estimated relative pose
        """
        # Estimate relative pose
        R, t, mask_pose = estimate_pose(kp1, kp2, matches, K)

        # Initialize cameras
        cam1 = Camera(K, np.eye(3), np.zeros(3), kp=kp1)
        cam2 = Camera(K, R, t, kp=kp2)
        self.add_camera(cam1, cam1_idx)
        self.add_camera(cam2, cam2_idx)

        # Triangulate initial points
        self.triangulate_new_points(cam1_idx, cam2_idx, kp1, kp2, matches, N=N, min_depth=min_depth, max_depth=max_depth)

        return R, t, mask_pose

    def filter_points_by_reprojection(self, thresh=5.0):
        """
        Reproject all 3D points into their observing cameras and
        remove points with large reprojection error.

        Args:
            thresh (float): Max reprojection error in pixels.
        """
        kept_points = {}
        removed = 0

        for pt3d in self.points3D.values():
            errors = []
            for img_idx, kp_idx in pt3d.observations.items():
                cam = self.cameras[img_idx]
                P = cam.projection_matrix()

                X_h = np.hstack([pt3d.coord, 1.0])
                x_proj = P @ X_h
                x_proj /= x_proj[2]
                u, v = x_proj[0], x_proj[1]

                kp = cam.kp[kp_idx].pt
                err = np.linalg.norm(np.array([u, v]) - np.array(kp))
                errors.append(err)

            if all(e < thresh for e in errors):
                kept_points[pt3d.id] = pt3d
            else:
                removed += 1

        self.points3D = kept_points
        print(f"[Reprojection filter] Removed {removed} points, kept {len(self.points3D)}.")
    
    # This version is slower

    # def filter_points_by_spatial_outliers(self, k=5, z_thresh=3.0):
    #     """
    #     Remove points that are spatial outliers using nearest-neighbor distances.

    #     Args:
    #         k (int): number of nearest neighbors to check
    #         z_thresh (float): threshold in z-score units
    #     """
    #     if len(self.points3D) < k + 1:
    #         return

    #     pts = np.array([p.coord for p in self.points3D.values()])
    #     # Computer pairwise distances
    #     dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    #     np.fill_diagonal(dists, np.inf)
    #     knn_dist = np.partition(dists, k, axis=1)[:, :k].mean(axis=1)
        
    #     #Z-score filtering
    #     mean_d, std_d = np.mean(knn_dist), np.std(knn_dist)
    #     z_scores = (knn_dist - mean_d) / (std_d + 1e-8)

    #     keep_mask = z_scores < z_thresh
    #     removed = 0
    #     new_points = {}
    #     for keep, p in zip(keep_mask, list(self.points3D.values())):
    #         if keep:
    #             new_points[p.id] = p
    #         else:
    #             removed += 1
    #     self.points3D = new_points
    #     print(f"[Spatial filter] removed {removed} outlier points")

    def filter_points_by_spatial_outliers(self, k=5, z_thresh=3.0):
        """
        Remove points that are spatial outliers using nearest-neighbor distances,
        but in a memory-efficient way using a KD-tree.

        Args:
            k (int): number of nearest neighbors to check
            z_thresh (float): threshold in z-score units
        """
        if len(self.points3D) < k + 1:
            return

        pts = np.array([p.coord for p in self.points3D.values()])
        tree = cKDTree(pts)

        # query k+1 nearest neighbors (including self)
        dists, _ = tree.query(pts, k=k+1)
        # exclude the self-distance (0)
        knn_dist = dists[:, 1:].mean(axis=1)

        # Z-score filtering
        mean_d, std_d = np.mean(knn_dist), np.std(knn_dist)
        z_scores = (knn_dist - mean_d) / (std_d + 1e-8)

        keep_mask = z_scores < z_thresh
        removed = 0
        new_points = {}
        for keep, p in zip(keep_mask, list(self.points3D.values())):
            if keep:
                new_points[p.id] = p
            else:
                removed += 1
        self.points3D = new_points
        print(f"[Spatial filter] removed {removed} outlier points")

    def merge_points_from_close_cameras(self, match_data, min_matches=200):
        """
        Merge 3D points using matches between cameras with strong overlap.
        Only considers camera pairs already registered in the reconstruction.
        """
        for key, matches in match_data.items():
            if len(matches) < min_matches:
                print(f"Skipping the rest of matches...")
                break

            i, j = map(int, key.split("_"))
            cam_i = self.cameras.get(i, None)
            cam_j = self.cameras.get(j, None)
            if cam_i is None or cam_j is None:
                continue

            for m in matches:
                kp_i, kp_j = m[0], m[1]

                pt_i = cam_i.observations.get(kp_i, None)
                pt_j = cam_j.observations.get(kp_j, None)

                if pt_i is not None and pt_j is None:
                    pt_i.observations[j] = kp_j
                    cam_j.observations[kp_j] = pt_i

                elif pt_i is None and pt_j is not None:
                    pt_j.observations[i] = kp_i
                    cam_i.observations[kp_i] = pt_j

                elif pt_i is not None and pt_j is not None and pt_i.id != pt_j.id:
                    # merge pt_j into pt_i
                    for cam_idx, kp_idx in pt_j.observations.items():
                        pt_i.observations[cam_idx] = kp_idx
                        self.cameras[cam_idx].observations[kp_idx] = pt_i
                    # delete old point
                    if pt_j.id in self.points3D:
                        del self.points3D[pt_j.id]

