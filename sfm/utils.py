import exifread
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from .features import deserialize_keypoints

# small lookup table for common cameras (you can expand this)
CAMERA_SENSOR_WIDTHS_MM = {
    "Panasonic DMC-GF6": 17.3,
    "Panasonic DMC-GF7": 17.3,
    "Olympus E-M10": 17.3,
    "Google Pixel 9a": 6.4                             #My phone :D
}

CAMERA_FOCAL_LENGTHS_MM = {
    "Google Pixel 9a": 4.53                             #Video mode                             
}

def get_exif_from_file(img_filepath):
    """
    Extract important EXIF data from an image file.

    Args:
        filepath (str): Path to the image file.

    Returns:
        dict: Dictionary containing:
            - 'focal_mm' (float or None): Focal length in mm.
            - 'width_px' (int or None): Image width in pixels.
            - 'height_px' (int or None): Image height in pixels.
    """
    with open(img_filepath, "rb") as f:
        tags = exifread.process_file(f)
    
    # Uncomment this if you want to see *all* metadata
    # for k, v in tags.items():
    #     print(f"{k}: {v}")

    # Extract tags
    focal = tags.get("EXIF FocalLength")
    width = tags.get("EXIF ExifImageWidth")
    height = tags.get("EXIF ExifImageLength")
    focal_35 = tags.get("EXIF FocalLengthIn35mmFilm")
    make = tags.get("Image Make")
    model = tags.get("Image Model")

    # Convert to float/int safely
    focal_mm = float(focal.values[0]) if focal is not None else None
    width_px = int(str(width)) if width is not None else None
    height_px = int(str(height)) if height is not None else None
    focal_35_mm = float(focal_35.values[0]) if focal_35 is not None else None

    make_s = str(make).strip() if make else None
    model_s = str(model).strip() if model else None
    model = make_s + " " + model_s if model_s and make_s else None

    return {
        "focal_mm": focal_mm,
        "focal_35_mm": focal_35_mm,
        "width_px": width_px,
        "height_px": height_px,
        "model": model
    }

import cv2
import numpy as np

def build_intrinsic_matrix(img_filepath, sensor_width=None, model=None):
    """
    Build the intrinsic camera matrix K from EXIF data (if available) or image size.

    Args:
        img_filepath (str): Path to the image file.
        sensor_width (float or None): Sensor width in mm. If None, will try model or fallback heuristic.
        model (str or None): Camera model name. Overrides EXIF model if provided.

    Returns:
        np.ndarray: 3x3 intrinsic matrix K
    """
    print("================ Building K (Intrinsic) matrix ================")
    
    exif_data = get_exif_from_file(img_filepath)
    
    # Use EXIF model if user didn't pass a model
    exif_model = exif_data.get("model")
    model = model or exif_model

    f_mm = exif_data.get("focal_mm")
    w_px = exif_data.get("width_px")
    h_px = exif_data.get("height_px")
    f_35_mm = exif_data.get("focal_35_mm")

    # Fallback: read image if EXIF size missing
    if w_px is None or h_px is None:
        img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {img_filepath}")
        h_px, w_px = img.shape[:2]

    # Try to infer sensor width and focal length from model if available
    if model:
        print(f"Using camera model: {model}")
        if sensor_width is None and model in CAMERA_SENSOR_WIDTHS_MM:
            sensor_width = CAMERA_SENSOR_WIDTHS_MM[model]
            print(f"Found sensor width for {model}: {sensor_width} mm")
        if f_mm is None and model in CAMERA_FOCAL_LENGTHS_MM:
            f_mm = CAMERA_FOCAL_LENGTHS_MM[model]
            print(f"Found focal length for {model}: {f_mm} mm")

    # Compute focal length in pixels
    if f_mm is not None and sensor_width is not None:
        f_px = f_mm * (w_px / sensor_width)
        print(f"Computed f_px from f_mm & sensor_width: {f_px:.2f}")
    elif f_35_mm is not None:
        diag_px = (w_px**2 + h_px**2)**0.5
        diag_35 = (36**2 + 24**2)**0.5  # 35mm full-frame diagonal
        f_px = f_35_mm * (diag_px / diag_35)
        print(f"Computed f_px from focal_35mm: {f_px:.2f}")
    else:
        # Fallback heuristic
        f_px = 1.2 * max(w_px, h_px)
        print(f"[WARN] Using heuristic for f_px: {f_px:.2f}")

    # Principal point (image center)
    cx = w_px / 2.0
    cy = h_px / 2.0

    K = np.array([
        [f_px, 0,   cx],
        [0,    f_px, cy],
        [0,    0,    1]
    ], dtype=np.float32)

    return K

def load_images(image_paths):
    """Load images in grayscale."""
    return [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]

def resize_keypoints(kps, scale):
    """
    Resize keypoints to be able to plot them with scaled images
    """
    new_kps = []
    for kp in kps:
        new_kps.append(cv2.KeyPoint(
            kp.pt[0] * scale,
            kp.pt[1] * scale,
            kp.size * scale,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        ))
    return new_kps

def plot_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    """
    Visualize feature matches between two images using matplotlib.

    Args:
        img1 (numpy.ndarray): First input image (grayscale or BGR)
        kp1 (list of cv2.KeyPoint): Detected keypoints in the first image
        img2 (numpy.ndarray): Second input image (grayscale or BGR)
        kp2 (list of cv2.KeyPoint): Detected keypoints in the second image
        matches (list of cv2.DMatch): Matches between kp1 and kp2
        max_matches (int, optional): Maximum number of matches to display

    Returns:
        None. Displays a matplotlib figure with:
            - img1 and img2 placed side by side,
            
            - matched keypoints highlighted (red in img1, green in img2),
            - yellow lines connecting matched keypoints across the images.
    """
    # compute scale factor
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    scale = 800 / max(w1, w2)

    # resize images
    img1_small = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
    img2_small = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))

    # scale keypoints
    kp1_small = resize_keypoints(kp1, scale)
    kp2_small = resize_keypoints(kp2, scale)

    # draw matches directly on smaller images
    img_matches = cv2.drawMatches(
        img1_small, kp1_small,
        img2_small, kp2_small,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_multiple_matches(image_files, all_kp, all_matches, max_pairs=9, max_matches=30, grid_cols=3):
    """
    Plot multiple image pair matches in a grid.

    Args:
        image_files (list): List of image paths.
        all_kp (dict): Serialized keypoints {img_idx: list of cv2.KeyPoint}.
        all_matches (dict): Dictionary { "i_j": np.array([[queryIdx, trainIdx], ...]) }.
        max_pairs (int): Maximum number of image pairs to plot.
        max_matches (int): Maximum number of matches to display per pair.
        grid_cols (int): Number of columns in the grid (default 3).
    """
    # pick top pairs
    selected_pairs = list(all_matches.items())[:max_pairs]
    n_pairs = len(selected_pairs)
    grid_rows = (n_pairs + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 5 * grid_rows))
    axes = axes.ravel() if n_pairs > 1 else [axes]

    for idx, (pair_key, matches_array) in enumerate(selected_pairs):
        i, j = map(int, pair_key.split("_"))
        img1 = cv2.imread(image_files[i], cv2.IMREAD_COLOR)
        img2 = cv2.imread(image_files[j], cv2.IMREAD_COLOR)

        kp1 = deserialize_keypoints(all_kp[i])
        kp2 = deserialize_keypoints(all_kp[j])

        # convert np array of indices into cv2.DMatch list
        matches = [cv2.DMatch(_queryIdx=q, _trainIdx=t, _imgIdx=0, _distance=0)
                   for q, t in matches_array[:max_matches]]

        # draw matches
        img_matches = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"Pair {i}-{j} ({len(matches_array)} inliers)")
        axes[idx].axis("off")

    # hide empty subplots if grid is larger than pairs
    for k in range(n_pairs, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()

def create_camera_frustum(K, R=np.eye(3), t=np.zeros((3,1)), 
                          image_size=(640,480), scale=0.1, color=[0,0,1]):
    """
    Create a camera frustum as an Open3D LineSet.
    
    K : 3x3 intrinsic matrix
    R : 3x3 rotation (world-to-camera or camera-to-world, see note)
    t : 3x1 translation (same convention as R)
    image_size : (width, height)
    scale : size scaling for visualization
    color : RGB list
    
    Returns: open3d.geometry.LineSet
    """
    w, h = image_size
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Define corners in pixel space
    corners_px = np.array([
        [0,   0],
        [w,   0],
        [w,   h],
        [0,   h]
    ])

    # Back-project to normalized camera coords (z=1 plane)
    corners_cam = np.array([[(x - cx) / fx, (y - cy) / fy, 1.0] 
                            for x, y in corners_px])

    # Scale frustum
    corners_cam *= scale
    cam_center = np.array([[0,0,0]])

    # Transform to world coords (here R,t are from cam-to-world)
    Rt = np.hstack((R, t.reshape(3,1)))
    cam2world = np.vstack((Rt, [0,0,0,1]))

    corners_h = np.hstack((corners_cam, np.ones((4,1))))
    cam_center_h = np.hstack((cam_center, np.ones((1,1))))

    corners_world = (cam2world @ corners_h.T).T[:,:3]
    cam_center_world = (cam2world @ cam_center_h.T).T[:,:3]

    # Adjust the distance of the corners
    frustum_depth = 0.5  # distance from camera center
    corners_cam = corners_cam / corners_cam[:,2].reshape(-1,1) * frustum_depth
    
    # Build line connections
    points = np.vstack((cam_center_world, corners_world))
    lines = [
        [0,1],[0,2],[0,3],[0,4],  # cam_center to corners
        [1,2],[2,3],[3,4],[4,1]   # frustum edges
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(color)
    return line_set

def create_camera_with_text(K, R=np.eye(3), t=np.zeros((3,1)),
                            image_size=(640,480), scale=0.1, color=[0,0,1],
                            text=None, text_scale=0.01, text_offset=np.array([0,0,0.04]),
                            text_color=[1,0,0], frustum_depth=0.5):
    """
    Create a camera frustum with optional 3D text label using precise intrinsics.
    Text is placed in the same coordinate frame as the frustum.
    """
    geometries = []

    w, h = image_size
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Define corners in pixel space
    corners_px = np.array([
        [0,   0],
        [w,   0],
        [w,   h],
        [0,   h]
    ])

    # Back-project to normalized camera coords (z=1 plane)
    corners_cam = np.array([[(x - cx) / fx, (y - cy) / fy, 1.0] 
                            for x, y in corners_px])

    # Scale frustum
    corners_cam *= scale
    cam_center = np.array([[0,0,0]])

    # Transform to world coords (R, t from cam-to-world)
    Rt = np.hstack((R, t.reshape(3,1)))
    cam2world = np.vstack((Rt, [0,0,0,1]))

    corners_h = np.hstack((corners_cam, np.ones((4,1))))
    cam_center_h = np.hstack((cam_center, np.ones((1,1))))

    corners_world = (cam2world @ corners_h.T).T[:,:3]
    cam_center_world = (cam2world @ cam_center_h.T).T[:,:3]

    # Build line connections
    points = np.vstack((cam_center_world, corners_world))
    lines = [
        [0,1],[0,2],[0,3],[0,4],  # cam_center to corners
        [1,2],[2,3],[3,4],[4,1]   # frustum edges
    ]

    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    frustum.paint_uniform_color(color)
    geometries.append(frustum)

    # Add text
    if text is not None:
        # Place the text slightly in front of the camera, aligned with frustum
        # Compute the text position as average of corners + offset
        text_pos = corners_world.mean(axis=0) + text_offset

        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.05).to_legacy()
        text_mesh.paint_uniform_color(text_color)

        # Center and scale text
        text_mesh.translate(-text_mesh.get_center())
        text_mesh.scale(text_scale, center=np.zeros(3))

        # Move to computed position
        T = np.eye(4)
        T[:3,3] = text_pos
        text_mesh.transform(T)

        geometries.append(text_mesh)

    return geometries

def save_reconstruction_points(reconstruction, filename="reconstruction.ply"):
    """
    Save the dictionary of points in reconstruction object to a ply file with Open3D
    """
    pts = np.array([p.coord for p in reconstruction.points3D.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {len(pts)} points to {filename}")

def save_cameras_np(reconstruction, filename="cameras.npz"):
    """
    Save camera parameters (K, R, t) to a .npz file.
    """
    cam_data = {}
    for cam_idx, cam in reconstruction.cameras.items():
        cam_data[str(cam_idx)] = {   # convert index to string
            "K": cam.K,
            "R": cam.R,
            "t": cam.t
        }
    np.savez(filename, **cam_data)
    print(f"Saved {len(reconstruction.cameras)} cameras to {filename}")

def load_cameras_np(filename="cameras.npz"):
    """
    Load camera parameters from a .npz file.
    Returns a dictionary of cam_idx (int) -> dict with K, R, t.
    """
    data = np.load(filename, allow_pickle=True)
    cameras = {int(k): v.item() for k, v in data.items()}
    return cameras

def visualize_sparse_cloud(reconstruction, scale=0.5, image_size=(640, 480)):
    """
    Visualize 3D points and cameras using Open3D.

    Args:
        reconstruction: Reconstruction object containing cameras and points3D
        scale: frustum size scaling
    """
    geometries = []

    pts = np.array([p.coord for p in reconstruction.points3D.values()])
    if len(pts) == 0:
        print("No points to visualize yet.")
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        geometries.append(pcd)

    # Add camera frustums
    for cam_idx, cam in reconstruction.cameras.items():
        frustum = create_camera_frustum(
            K=cam.K,
            R=cam.R,
            t=cam.t,
            image_size=image_size,
            color=[0, 0, 1],
            scale=scale
        )
        geometries.append(frustum)

    o3d.visualization.draw_geometries(geometries)
