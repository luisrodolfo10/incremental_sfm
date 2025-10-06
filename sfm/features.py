import cv2
import numpy as np

def detect_and_match(img1, img2, mask1=None, mask2=None, method="SIFT", matcher_type="BF", cross_check=False, ratio_thresh=0.75):
    """
    Detect keypoints and match features between two images.

    Args:
        img1, img2: Input images, usually grayscale is better.
        mask1, mask2: Input masks, 
        method: Feature detector ("SIFT", "ORB", "AKAZE").
        matcher_type: "BF" for Brute-Force, "FLANN" for FLANN-based matching.
        cross_check: If True, enables cross-check filtering for BFMatcher.
        ratio_thresh: Lowe's ratio threshold, 0.75 gives pretty decent results to fitler ambigous matches.
    Returns:
        kp1, kp2: Keypoints from both images.
        good_matches: List of filtered good matches.
    """

    # Initialize detector
    if method == "SIFT":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == "ORB":
        detector = cv2.ORB_create()
        norm_type = cv2.NORM_HAMMING
    elif method == "AKAZE":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown method: {method}")
    
    #Get keypoints and descriptors based on the method

    # Detect keypoints and compute descriptors (vectors that describe the local appearance around a keypoint)
    kp1, des1 = detector.detectAndCompute(img1, mask1)
    kp2, des2 = detector.detectAndCompute(img2, mask2)

    if matcher_type == "BF":
        bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)
        if cross_check: # Only keep matches that are mutual (Use it when there are not many matches)
            good_matches = bf.match(des1, des2)
            #Optional sorting for debugging
            #good_matches = sorted(good_matches, key=lambda x: x.distance) 
        else:
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
            #Optional sorting for debugging
            #good_matches = sorted(good_matches, key=lambda x: x.distance) 
        
    elif matcher_type == "FLANN":
        # Efficient for large datasets, uses approximate nearest neighbor search
        # instead of brute-force. Can handle both float and binary descriptors.
        if method == "SIFT":
            # For float descriptors (like SIFT), use KD-Tree algorithm
            # trees specifies number of parallel trees to use (higher = more accuracy, slower)
            index_params = dict(algorithm=1, trees=5)  # KDTree for SIFT
        else:
            # For binary descriptors (like ORB or AKAZE), use LSH (Locality Sensitive Hashing)
            # table_number, key_size, multi_probe_level are parameters controlling hash tables
            # and search accuracy vs. speed
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) 

        # The number the tree is recursively checked
        search_params = dict(checks=50)  # Higher = more accurate but slower

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 2 nearest neighbors for each descriptor
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

        #Optional sorting for debugging
        #good_matches = sorted(good_matches, key=lambda x: x.distance) 

    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")
    
    return kp1, kp2, good_matches


def detect_features(img, mask=None, method="SIFT", nfeatures=None):
    """
    Detect keypoints and descriptors for a single image.

    Args:
        img: Input grayscale image
        mask: Optional mask
        method: "SIFT", "ORB", "AKAZE"
    Returns:
        kp: list of cv2.KeyPoint
        des: descriptors as numpy array
    """
    if method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=nfeatures)
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif method == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown method: {method}")

    kp, des = detector.detectAndCompute(img, mask)
    return kp, des

def match_features(des1, des2, method="SIFT", matcher_type="BF", cross_check=False, ratio_thresh=0.75):
    """
    Match descriptors between two images.

    Args:
        des1, des2: descriptors from detect_features
        method: "SIFT", "ORB", "AKAZE"
        matcher_type: "BF" or "FLANN"
        cross_check: for BFMatcher
        ratio_thresh: Lowe's ratio test threshold
    Returns:
        good_matches: list of cv2.DMatch
    """
    # Norm type
    if method == "SIFT":
        norm_type = cv2.NORM_L2
    else:
        norm_type = cv2.NORM_HAMMING

    # BFMatcher
    if matcher_type == "BF":
        bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)
        if cross_check:
            good_matches = bf.match(des1, des2)
        else:
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    # FLANN
    elif matcher_type == "FLANN":
        if method == "SIFT":
            index_params = dict(algorithm=1, trees=5)
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")

    return good_matches

# Storing data
def serialize_keypoints(kp_list):
    """
    Convert cv2.KeyPoint list to numpy array for storage.
    """
    return np.array([(
        kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id
    ) for kp in kp_list], dtype=np.float32)

def deserialize_keypoints(kp_array):
    """
    Convert a stored array back to cv2.KeyPoint objects.
    
    kp_array: numpy array of shape (N, 7) with [x, y, size, angle, response, octave, class_id]
    """
    kp_list = []
    for x, y, size, angle, response, octave, class_id in kp_array:
        kp = cv2.KeyPoint(
            x=float(x),
            y=float(y),
            size=float(size),
            angle=float(angle),
            response=float(response),
            octave=int(octave),
            class_id=int(class_id)
        )
        kp_list.append(kp)
    return kp_list

def load_matches(match_data, pair_key, flip=False):
    """
    Load matches for a specific image pair from a .npz file.

    Args:
        match_data: The dictionary containing all the matches
        pair_key: Key corresponding to the image pair, e.g., "1_3"
        flip: If True, swap queryIdx and trainIdx
    Returns:
        matches: list of cv2.DMatch
    """
    if pair_key not in match_data:
        raise KeyError(f"Pair key {pair_key} not found in matches file.")
    
    #matches_array = match_data[pair_key]

    # Using pop function to remove the pair_key and not using it again
    matches_array = match_data.pop(pair_key)

    if flip:
        matches = [cv2.DMatch(_queryIdx=int(j), _trainIdx=int(i), _imgIdx=0, _distance=0)
                   for i, j in matches_array]
    else:
        matches = [cv2.DMatch(_queryIdx=int(i), _trainIdx=int(j), _imgIdx=0, _distance=0)
                   for i, j in matches_array]

    return matches

def compute_global_descriptor(img, method="ORB", n_features=500):
    # Downscale for speed
    img_small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    if method == "ORB":
        orb = cv2.ORB_create(nfeatures=n_features)
        kp, des = orb.detectAndCompute(img_small, None)
        # Aggregate ORB descriptors into one vector, e.g., mean
        if des is not None and len(des) > 0:
            descriptor = des.mean(axis=0)
        else:
            descriptor = np.zeros(32)  # ORB descriptor size
    elif method == "HIST":
        # Convert to HSV and compute color histogram
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        descriptor = hist.flatten()
    else:
        raise ValueError("Unknown method")
    return descriptor