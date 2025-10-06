import cv2
import numpy as np

def triangulate_points(kp1, kp2, matches, R, t, K, mask_pose=None):
    """
    Triangulate 3D points from two calibrated camera views.

    Inputs:
        kp1 (list of cv2.KeyPoint) - Keypoints from the first image.
        kp2 (list of cv2.KeyPoint) - Keypoints from the second image.
        matches (list of cv2.DMatch) - Good matches between kp1 and kp2.
        R (3x3 numpy.ndarray) - Rotation matrix from camera1 to camera2.
        t (3x1 numpy.ndarray) - Translation vector from camera1 to camera2 (scale unknown).
        K (3x3 numpy.ndarray) - Camera intrinsic calibration matrix:
            [ [fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1] ]
        mask_pose (numpy.ndarray) - Binary mask of inlier matches after pose recovery

    Outputs:
        pts3D (Nx3 numpy.ndarray) - Reconstructed 3D points in homogeneous coordinates
                                    converted to Euclidean coordinates.
    """
    if mask_pose is not None:
        matches = [m for m, inl in zip(matches, mask_pose.ravel()) if inl]
    #print(matches)

    # Extract the matched 2D point coordinates from both images
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Build the projection matrix for camera 1:
    # [R | t]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))

    # Build the projection matrix for camera 2:
    #P2 = K @ np.hstack((R, t))
    P2 = K @ np.hstack((R, t.reshape(3,1)))
    
    # Triangulate homogeneous 4D points (x, y, z, w)
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert from homogeneous coordinates (x, y, z, w) -> (x/w, y/w, z/w)
    pts3D = (pts4D[:3] / pts4D[3]).T

    return pts3D, matches

def triangulate_points_world_coords(kp1, kp2, matches, cam1, cam2):
    """
    Triangulate 3D points from two calibrated camera views 
    and with known camera poses and rotation to give points in World Coordinates

    Inputs:
        kp1 (list of cv2.KeyPoint) - Keypoints from the first image.
        kp2 (list of cv2.KeyPoint) - Keypoints from the second image.
        matches (list of cv2.DMatch) - Good matches between kp1 and kp2.
        cam1 - 1st cam instance of reconstruction - Will use its R, t and K for triangulation
        cam2 - 2nd cam instance of reconsrtruction - Will use its R, t and K for triangulation
    Outputs:
        pts3D (Nx3 numpy.ndarray) - Reconstructed 3D points in homogeneous coordinates
                                    converted to Euclidean coordinates.
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    P1 = cam1.K @ np.hstack((cam1.R, cam1.t.reshape(3,1)))
    P2 = cam2.K @ np.hstack((cam2.R, cam2.t.reshape(3,1)))

    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T

    return pts3D

def filter_by_reprojection(points3D, kp1, kp2, matches, R, t, K, thresh=2.0):
    if len(points3D) != len(matches):
        raise ValueError(f"Length mismatch: points3D={len(points3D)}, matches={len(matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Camera projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t.reshape(3,1)))

    # Reproject
    pts1_proj_h = (P1 @ np.hstack((points3D, np.ones((len(points3D),1)))).T).T
    pts2_proj_h = (P2 @ np.hstack((points3D, np.ones((len(points3D),1)))).T).T

    pts1_proj = (pts1_proj_h[:,:2].T / pts1_proj_h[:,2]).T
    pts2_proj = (pts2_proj_h[:,:2].T / pts2_proj_h[:,2]).T

    # Compute reprojection error
    err1 = np.linalg.norm(pts1 - pts1_proj, axis=1)
    err2 = np.linalg.norm(pts2 - pts2_proj, axis=1)
    mask = (err1 < thresh) & (err2 < thresh)

    # Filter points and matches consistently
    points3D_filtered = points3D[mask]
    matches_filtered = [m for i, m in enumerate(matches) if mask[i]]

    #print(f"Reprojection error: mean={err1.mean():.2f}/{err2.mean():.2f}, max={err1.max():.2f}/{err2.max():.2f}")


    return points3D_filtered, matches_filtered

def filter_by_depth(points3D, min_depth=0.1, max_depth=50.0, matches=None):
    z = points3D[:,2]
    mask = (z > min_depth) & (z < max_depth)
    points3D_filtered = points3D[mask]

    if matches is not None:
        matches_filtered = [m for i, m in enumerate(matches) if mask[i]]
        return points3D_filtered, matches_filtered
    else:
        return points3D_filtered


def filter_by_cheirality(points3D, R, t, matches=None):
    # Points in camera 1 frame (z > 0)
    mask1 = points3D[:,2] > 0
    # Transform to camera 2 frame
    pts_cam2 = (R @ points3D.T + t.reshape(3,1)).T
    mask2 = pts_cam2[:,2] > 0
    mask = mask1 & mask2

    points3D_filtered = points3D[mask]

    if matches is not None:
        matches_filtered = [m for i, m in enumerate(matches) if mask[i]]
        return points3D_filtered, matches_filtered
    else:
        return points3D_filtered
    
def filter_by_cheirality_world_coords(points3D, cam1, cam2, matches=None):
    """
    Filter 3D points to keep only those in front of both known cameras.

    Args:
        points3D (np.ndarray): Nx3 array of 3D points in world coordinates
        cam1, cam2 (Camera): Known cameras with R, t in world-to-camera frame
        matches (list of cv2.DMatch, optional): Corresponding matches

    Returns:
        points3D_filtered (np.ndarray): Filtered 3D points
        matches_filtered (list of cv2.DMatch, optional): Filtered matches
    """
    # Transform points to camera frames
    pts_cam1 = (cam1.R @ points3D.T + cam1.t.reshape(3,1)).T
    pts_cam2 = (cam2.R @ points3D.T + cam2.t.reshape(3,1)).T

    # Check positive depth
    mask = (pts_cam1[:,2] > 0) & (pts_cam2[:,2] > 0)
    points3D_filtered = points3D[mask]

    if matches is not None:
        matches_filtered = [m for i, m in enumerate(matches) if mask[i]]
        return points3D_filtered, matches_filtered
    else:
        return points3D_filtered
