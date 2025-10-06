import os
import cv2
import numpy as np
from sfm.features import (
    detect_features,
    match_features,
    serialize_keypoints,
    deserialize_keypoints,
    compute_global_descriptor
)
from sfm.utils import build_intrinsic_matrix, plot_multiple_matches
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    ######################################################################
    ######################### Paths and settings #########################
    dataset = "patito"
    image_dir = "data/images/" + dataset
    mask_dir = "data/masks/" + dataset
    feature_file = "data/features/" + dataset + "-geom.npz"      # Stored features

    method = "SIFT"        # (SIFT, AKAZE or ORB)
    matcher = "FLANN"      # (BF, FLANN)
    ratio = 0.75           # Lowe's ratio
    min_matches = 10       # Minimum inlier matches after geometric verification
    sort_matches = True
    limit_images = None    # Limit number of images
    nfeatures = None       # Number of detector features to use

    # pairing_mode can be: "descriptor", "sequential", "normal"
    pairing_mode = "normal"

    ######################################################################
    ########################### Load images ##############################
    image_files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    mask_files = None
    if os.path.exists(mask_dir):
        mask_files = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        assert len(mask_files) == len(image_files), "Number of masks must match number of images"

    if limit_images and limit_images < len(image_files):
        image_files = image_files[:limit_images]

    ######################################################################
    ####################### Build candidate pairs ########################
    if pairing_mode == "descriptor":
        print("Filtering match pairs based on global descriptors...")
        global_descs = []
        for img_fp in image_files:
            img = cv2.imread(img_fp)
            desc = compute_global_descriptor(img, method="ORB")
            global_descs.append(desc)
        global_descs = np.array(global_descs)

        similarity = cosine_similarity(global_descs)
        top_k = 5  # match each image with top 5 similar ones

        candidate_pairs = []
        for i in range(len(image_files)):
            neighbors = np.argsort(similarity[i])[::-1]  # descending
            for j in neighbors[1:top_k + 1]:  # skip self
                if i < j:  # avoid duplicates
                    candidate_pairs.append((i, int(j)))

        print(f"Descriptor-based pairing: {len(candidate_pairs)} pairs")

    elif pairing_mode == "sequential":
        print("Using sequential pairing (i with i+1).")
        candidate_pairs = [(i, i + 1) for i in range(len(image_files) - 1)]

    elif pairing_mode == "normal":
        print("Using normal pairing (all ordered pairs except self).")
        candidate_pairs = [(i, j) for i in range(len(image_files)) for j in range(len(image_files)) if i != j]

    else:
        raise ValueError("Invalid pairing_mode. Choose from: 'descriptor', 'sequential', 'global', or 'normal'.")

    print(f"Total candidate pairs: {len(candidate_pairs)}")
    ######################################################################

    K = build_intrinsic_matrix(image_files[0])  # Assume shared intrinsics

    ######################################################################
    ######################## Detect keypoints ############################
    all_kp = {}
    all_des = {}
    print("Detecting keypoints for all images...")
    for idx, img_fp in enumerate(image_files):
        img = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE) if mask_files else None
        kp, des = detect_features(img, mask=mask, method=method, nfeatures=nfeatures)
        all_kp[idx] = serialize_keypoints(kp)
        all_des[idx] = des
        print(f"Image {idx}: {len(kp)} keypoints")

    ######################################################################
    #################### Match + Geometric verification ##################
    all_matches = {}
    print("Matching image pairs with geometric verification...")
    for i, j in candidate_pairs:
        # Step 1: Descriptor matching
        good_matches = match_features(all_des[i], all_des[j], method=method,
                                      matcher_type=matcher, ratio_thresh=ratio)

        if len(good_matches) < min_matches:
            continue

        # Step 2: Geometric verification (Essential matrix + RANSAC)
        kp1 = deserialize_keypoints(all_kp[i])
        kp2 = deserialize_keypoints(all_kp[j])

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        if len(pts1) >= 5:  # Need at least 5 points for E
            E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
            if E is None:
                continue

            inlier_matches = [m for m, inl in zip(good_matches, mask.ravel()) if inl]

            if len(inlier_matches) >= min_matches:
                all_matches[f"{i}_{j}"] = np.array(
                    [[m.queryIdx, m.trainIdx] for m in inlier_matches],
                    dtype=np.int32
                )
                print(f"Pair {i}_{j}: {len(inlier_matches)} inliers / {len(good_matches)} matches")
            else:
                print(f"Pair {i}_{j}: only {len(inlier_matches)} inliers skipped")
        else:
            print(f"Pair {i}_{j}: not enough points for Essential matrix")

    ######################################################################
    ########################## Save and display ##########################
    if sort_matches:
        all_matches = dict(sorted(all_matches.items(), key=lambda x: len(x[1]), reverse=True))

    print("Top image pairs by number of inliers:")
    for k, v in list(all_matches.items())[:20]:
        print(f"{k}: {len(v)} inliers")

    np.savez(feature_file, keypoints=all_kp, matches=all_matches)
    print(f"Saved geometrically verified features to {feature_file}")

    # Load back and visualize
    data = np.load(feature_file, allow_pickle=True)
    loaded_kp = data["keypoints"].item()
    loaded_matches = data["matches"].item()

    print(f"Loaded {len(loaded_kp)} keypoint sets, {len(loaded_matches)} match sets")

    plot_multiple_matches(image_files, loaded_kp, loaded_matches, max_pairs=9, max_matches=40, grid_cols=3)
