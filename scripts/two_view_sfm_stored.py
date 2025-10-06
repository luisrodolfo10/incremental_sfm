import os
import cv2
import numpy as np
import open3d as o3d
from sfm.features import deserialize_keypoints, load_matches
from sfm.geometry import estimate_pose
from sfm.triangulation import triangulate_points, filter_by_depth, filter_by_reprojection
from sfm.utils import build_intrinsic_matrix, plot_matches, create_camera_frustum

# Script to generate a two_view sfm using stored data in features/
if __name__ == "__main__":
    # Paths
    image_dir = "data/images/patito"
    feature_file = "data/features/patito-geom.npz"

    # image_dir = "data/images/patito"
    # feature_file = "data/features/patito.npz"

    imgs_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_files = [os.path.join(image_dir, f) for f in imgs_list]

    # Build the intrinsci matrix using the first image
    K = build_intrinsic_matrix(image_files[0])
    #K = build_intrinsic_matrix(image_files[0])
    #K = build_intrinsic_matrix(image_files[0], exif=True, sensor_width=6.4)  #My phone's camera sensor width is 6.4mm (pixel 9a)

    # Load precomputed keypoints
    data = np.load(feature_file, allow_pickle=True)
    kp_data = data['keypoints'].item()
    print("Keypoints arrays from images: ", len(kp_data))
    match_data = data['matches'].item()
    print("Matches arrays from images: ", len(match_data))
    for match in match_data.values():
        print(len(match))

    # Select the first pair with the most matches
    pair_key = list(match_data.keys())[5]  # assumes already sorted descending by number of matches
    #pair_key = "58_60"
    print(match_data.keys())
    i, j = map(int, pair_key.split('_'))
    print(f"Using image pair {i} and {j} with {len(match_data[pair_key])} matches")

    kp1 = deserialize_keypoints(kp_data[i])
    kp2 = deserialize_keypoints(kp_data[j])

    print(f"len of kp {i}: {len(kp1)}")
    print(f"len of kp {j}: {len(kp2)}")

    # Load matches
    matches = load_matches(match_data, pair_key)

    # Load images
    img1 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_files[j], cv2.IMREAD_GRAYSCALE)

    # Visualize matches
    plot_matches(img1, kp1, img2, kp2, matches, max_matches=300)

    # Estimate pose
    R, t, mask_pose = estimate_pose(kp1, kp2, matches, K)

    # Triangulate points
    points3D, matches_inlier = triangulate_points(kp1, kp2, matches, R, t, K)

    # Filter points
    print("Before filtering: ", len(points3D))
    points3D, matches_inlier = filter_by_reprojection(points3D, kp1, kp2, matches_inlier, R, t, K)
    print("after reprojection: ", len(points3D))
    points3D, matches_inlier = filter_by_depth(points3D, 0.1, 15, matches=matches_inlier)

    # Visualize pointcloud and cameras
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)

    cam1 = create_camera_frustum(K, R=np.eye(3), t=np.zeros((3,1)), image_size=img1.shape[::-1], color=[1,0,0], scale=0.4)
    cam2 = create_camera_frustum(K, R=R, t=t, image_size=img2.shape[::-1], color=[0,1,0], scale=0.4)

    o3d.visualization.draw_geometries([pcd, cam1, cam2])
