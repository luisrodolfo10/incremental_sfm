import os
import cv2
import numpy as np
from sfm.features import detect_features, match_features, serialize_keypoints
from sfm.utils import plot_multiple_matches

if __name__ == "__main__":
    # -------------- Paths and settings -------------- #
    dataset = "cat"
    image_dir = "data/images/"+ dataset
    mask_dir = "data/masks/" + dataset 
    feature_file = "data/features/" + dataset + ".npz"      # Stored features

    #(SIFT, AKAZE or ORB) bmethod to detect features
    method = "SIFT" 
    #(BF, FLANN) matcher type - FLANN works great for large datasets
    matcher = "FLANN"
    
    ratio  = 0.75                           # Lowe's ratio for matching filtering
    min_matches = 30                        # Minimum number of matches to keep a pair
    sort_matches = True                     # If true the matches are sorted by number of matches (Larger > Smaller)
    limit_images = 40                     # Limit number of images, it can get huge

    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = None
    if os.path.exists(mask_dir):
        mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(mask_files) == len(image_files), "Number of masks must match number of images"

    # Limit the image files:
    if limit_images and limit_images < len(image_files):
        image_files = image_files[:limit_images]

    # Detect features
    all_kp = {}
    all_des = {}

    print("Detecting keypoints for all images...")
    for idx, img_fp in enumerate(image_files):
        img = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE) if mask_files else None
        kp, des = detect_features(img, mask=mask, method=method)
        all_kp[idx] = serialize_keypoints(kp)
        all_des[idx] = des
        print(f"Image {idx}: {len(kp)} keypoints")

    # Match features for all pairs
    all_matches = {}
    print("Matching all image pairs...")
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            good_matches = match_features(all_des[i], all_des[j], method=method,
                                        matcher_type=matcher, ratio_thresh=ratio)
            if len(good_matches) >= min_matches:
                # store as index pairs
                all_matches[f"{i}_{j}"] = np.array([[m.queryIdx, m.trainIdx] for m in good_matches], dtype=np.int32)
                print(f"Matched image {i} and {j}: {len(good_matches)} matches ")
            else:
                print(f"Matched image {i} and {j}: {len(good_matches)} matches skipped ( < {min_matches})")

    # Sort all_matches by number of matches (descending)
    all_matches = dict(sorted(all_matches.items(), key=lambda x: len(x[1]), reverse=True))
    print("Top image pairs by number of matches:")
    for k, v in list(all_matches.items())[:20]:
        print(f"{k}: {len(v)} matches")

    # Save everything
    np.savez(feature_file, keypoints=all_kp, matches=all_matches)
    print(f"Saved keypoints and matches for all images to {feature_file}")

    #  Load back example
    data = np.load(feature_file, allow_pickle=True)
    loaded_kp = data['keypoints'].item()
    print("Keypoints lists from images: ", len(loaded_kp))
    loaded_matches = data['matches'].item()
    print("Matches arrays from images: ", len(loaded_matches))

    #Plot matches
    plot_multiple_matches(image_files, loaded_kp, loaded_matches, max_pairs=9, max_matches=40, grid_cols=3)