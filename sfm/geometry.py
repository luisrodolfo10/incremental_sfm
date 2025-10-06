import cv2
import numpy as np

def estimate_pose(kp1, kp2, matches, K):
    """
    Estimate relative camera pose (rotation and translation) from matched keypoints.

    Inputs:
        kp1 (list of cv2.KeyPoint) - Keypoints from the first image
        kp2 (list of cv2.KeyPoint) - Keypoints from the second image
        matches (list of cv2.DMatch) - Good matches between kp1 and kp2
        K (numpy.ndarray, 3x3) - Camera intrinsic calibration matrix:
            [ [fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1] ]

    Outputs:
        R (3x3 numpy.ndarray) - Rotation matrix from camera1 to camera2
        t (3x1 numpy.ndarray) - Translation vector (unit length, scale unknown)
        mask_pose (numpy.ndarray) - Binary mask of inlier matches after pose recovery
    """
    # Extract the matched 2D point coordinates from both images
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
 
    # Estimates the Essential Matrix  between two views and get R and t from E using cheiraility check.
    #E, mask_E = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K) #Cheirality check (Points must be in fron of the camera)
    return R, t, mask_pose