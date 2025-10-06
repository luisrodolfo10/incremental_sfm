import os
import cv2
import numpy as np
import open3d as o3d
import threading
import time
from open3d.visualization import gui
from sfm.reconstruction import Reconstruction, Camera, Point3D
from sfm.features import deserialize_keypoints, load_matches
from sfm.bundle_adjustment import local_bundle_adjustment, global_bundle_adjustment
from sfm.utils import build_intrinsic_matrix, visualize_sparse_cloud, save_reconstruction_points, save_cameras_np
from sfm.gui import SfMVisualizer, SfMVisualizer_simple
import pandas as pd


############################################################
############### Settings for incremental SfM ###############
BUNDLE_N_CAMS = 3                           # Number of cams for local bundle adjustment
GLOBAL_BA_EVERY = 5                         # Run global cleanup every 5 cameras
LOOSE_REPROJ_THRESH = 40.0                  # Loose pixel error to filter obvious reprojection errors  
STRICT_REPROJ_THRESH = 7.0                  # Strict pixel error to filter reprojection errors at the end  (7)        
F_SCALE_LOCAL = 20.0                        # Tunning f_scale for scipy otpimzers
F_SCALE_GLOBAL = 5.0                        # Tunning f_scale for scipy optimizers
X_TOL = 1e-5                                # Tolerancies for Scipyt Optimize Least Squares
F_TOL = 1e-5
G_TOL = 1e-5
N_POINTS = 100                               # Limit of N points for each triangulation (50 for big ones) 
MIN_MATCHES = 50                           # Min number of matches to merge points/observations (100, big keypoints/matches dataset)
MIN_DEPTH = 0.1                             # Min depth for depth filtering while triangulating new points
MAX_DEPTH = 20                              # Max depth for depth filtering while triangulating new points (10-20 indoor, 100-150 outdoor)
############################################################

# -------------------------------
# Incremental SfM using stored keypoints & matches
# -------------------------------
def incremental_sfm(N_IMGS, feature_file, K, vis=None):
    stats_log = []
    reconstruction = Reconstruction()

    # Load precomputed keypoints and matches
    data = np.load(feature_file, allow_pickle=True)
    kp_data = data['keypoints'].item()
    match_data = data['matches'].item()

    # Deserialized keypoints
    kp_data = [deserialize_keypoints(kp_data[i]) for i in range(len(kp_data))]

    print("Keypoints arrays from images:", len(kp_data))
    print("Matches arrays from images:", len(match_data))

    # Initialize with first pair (most matches), assuming the stored matches are already sorted like in the extract_and_match script
    first_pair = list(match_data.keys())[0]
    i, j = map(int, first_pair.split("_"))
    print(f"Initializing with pair {first_pair}, {len(match_data[first_pair])} matches")
    kp1 = kp_data[i]
    kp2 = kp_data[j]
    used_pairs = {first_pair:match_data[first_pair]}            #Store used_pairs for fine reconstruction
    matches = load_matches(match_data, first_pair)
    R, t, mask_pose = reconstruction.run_two_view_init(kp1, kp2, matches, K, i, j, N=N_POINTS, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)

    used_images = [i, j]
    
    #plot_matches(cv2_imgs[i], kp1, cv2_imgs[j], kp2, matches, max_matches=300)
    points = [pt.coord for pt in reconstruction.points3D.values()]
    cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
    info = {"iteration": 0, "points": len(points), "cameras": len(cameras), "error": 0.0, 
            "time": 0.0, "pair": first_pair, "active_cams": [i, j], "status": "Intializing" }
    stats_log.append(info)
    if vis is not None:
        vis.post_update(points, cameras, info=info)
    
    start = time.time()        
    # Incrementally add remaining images
    while len(used_images) < N_IMGS:
        next_img = None
        base_img = None
        next_pair = None
        flip = False                    # When the query idx and train idx are swapped
        success = False                 # When there are enough points for PnP

        # Look for next image with most matches to already reconstructed cameras
        for key in match_data.keys(): 
            img_a, img_b = map(int, key.split("_"))
            if img_a in used_images and img_b not in used_images:
                next_img = img_b
                base_img = img_a
                break

            #Query and train order will be changed since based img is in the second index of the pair key
            elif img_b in used_images and img_a not in used_images:
                next_img = img_a
                base_img = img_b
                flip = True
                break

        #print(f"length of match data keys {len(match_data.keys())}")

        # The key is the next pair with the most matches
        next_pair=key

        if next_img is None:
            print("No more images can be registered")
            print(used_images)
            print(match_data.keys())
            break

        print(f"Adding pair {next_pair}, {len(match_data[next_pair])} matches")
        kp_next = kp_data[next_img]
        kp_base = kp_data[base_img]
        used_pairs[next_pair] = match_data[next_pair]
        matches = load_matches(match_data, next_pair, flip=flip)


        # Build 2D–3D correspondences for PnP (Which constructed 3D points can also be seen by the next camera)
        object_points, image_points = [], []

        # Debug plot to see matches
        #plot_matches(cv2_imgs[base_img], kp_base, cv2_imgs[next_img], kp_next, matches, max_matches=300)
        
        for m in matches:
            kp_idx_base = m.queryIdx
            kp_idx_next = m.trainIdx

            # Check if base camera already has this keypoint linked to a 3D point (checks keys)
            if kp_idx_base in reconstruction.cameras[base_img].observations:
                pt3d = reconstruction.cameras[base_img].observations[kp_idx_base]

                # Add correspondence
                object_points.append(pt3d.coord)
                image_points.append(kp_next[kp_idx_next].pt)
        
        object_points = np.array(object_points, dtype=np.float32).reshape(-1, 3)
        image_points  = np.array(image_points, dtype=np.float32).reshape(-1, 2)

        print(f"Points with 3D-2D correspondence: {len(object_points)}")

        if len(object_points) >= 4:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, K, None)
            R, _ = cv2.Rodrigues(rvec)

            cam_new = Camera(K, R, tvec)
            cam_new.kp = kp_next
            reconstruction.add_camera(cam_new, next_img)
            used_images.append(next_img)
            success = True
            print(f"PnP pass for: {next_img} from pair: {next_pair}")
        else:
            print(f"Not enough object points to perform PnP ({len(object_points)} points) skipping camera: {next_img} from pair: {next_pair}")
            del used_pairs[next_pair] # Not using this pair for fine reconstrucction
            continue

        if success:
            # Triangulate new points
            reconstruction.triangulate_new_points(base_img, next_img, kp_base, kp_next, matches, N=N_POINTS, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
            print(f"Triangulated new points from pair {next_pair}")

            # Local BA on recent cameras
            if len(used_images) >= BUNDLE_N_CAMS:
                local_cams = used_images[-BUNDLE_N_CAMS:]
                points = [pt.coord for pt in reconstruction.points3D.values()]
                cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
                info = {"iteration": len(used_images),"points": len(points),"cameras": len(cameras), "error": 0.0,
                        "time": time.time() - start,"pair": next_pair,"active_cams": local_cams,"status": f"Local BA (last {BUNDLE_N_CAMS} cams)" }
                stats_log.append(info)
                # Updating visualizer
                if vis is not None:
                    vis.post_update(points, cameras, info=info)

                start = time.time() #Count BA time
                print(f"Local bundle adjustment on last {BUNDLE_N_CAMS} cameras starting for camera {next_img}...")
                res = local_bundle_adjustment(reconstruction, K, local_cam_idxs=local_cams, verbose=2, 
                                        f_scale=F_SCALE_LOCAL, xtol=X_TOL, ftol=F_TOL, gtol=G_TOL)
                print(f"Local bundle adjustment on last {BUNDLE_N_CAMS} cameras done for camera {next_img}.")

                #print(f"Res: {res}")
                # RMS error from residuals
                if hasattr(res, "fun"):
                    rms_error = np.sqrt(np.mean(res.fun**2))
                    residuals = res.fun.reshape(-1, 2)
                    rms_error_per_obs = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
                else:
                    rms_error = -1

                points = [pt.coord for pt in reconstruction.points3D.values()]
                cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
                info = {"iteration": len(used_images), "points": len(points), "cameras": len(cameras), "error": rms_error, 
                        "time": time.time() - start, "pair": next_pair, "active_cams": local_cams, "status": f"Local BA (last {BUNDLE_N_CAMS} cams)"}
                stats_log.append(info)
                # Updating visualizer
                if vis is not None:
                    vis.post_update(points, cameras, info=info)
                    
            # Optionally: run reprojection filter and global BA every GLOBAL_BA_EVERY cameras
            if len(used_images) % GLOBAL_BA_EVERY == 0:
                points = [pt.coord for pt in reconstruction.points3D.values()]
                cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
                info = {"iteration": len(used_images),"points": len(points),"cameras": len(cameras), "error": rms_error,
                        "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Filtering: Loose reprojection"}
                stats_log.append(info)
                if vis is not None:
                    vis.post_update(points, cameras, global_flag=True, info=info)

                print(f"Running [Loose Reprojection filter] on all points (thresh={LOOSE_REPROJ_THRESH}px) ...")
                reconstruction.filter_points_by_reprojection(thresh=LOOSE_REPROJ_THRESH)
                points = [pt.coord for pt in reconstruction.points3D.values()]
                cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
                info = {"iteration": len(used_images),"points": len(points),"cameras": len(cameras), "error": rms_error, 
                        "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Filtering: Spatial Outliers"}
                stats_log.append(info)
                # Updating visualizer
                if vis is not None:
                    vis.post_update(points, cameras, global_flag=True, info=info)

                print("Running [Spatial Outlier filter] on all points ...")
                reconstruction.filter_points_by_spatial_outliers(k=5, z_thresh=3.0)

                # Updating visualizer
                points = [pt.coord for pt in reconstruction.points3D.values()]
                cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
                info = {"iteration": len(used_images),"points": len(points),"cameras": len(cameras),"error": rms_error,"time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Merging points/observations"}
                stats_log.append(info)
                if vis is not None:
                    vis.post_update(points, cameras, global_flag=True, info=info)

                print("Merging close-camera points (keypoint-based)...")
                reconstruction.merge_points_from_close_cameras(match_data, min_matches=MIN_MATCHES)

                 # Updating visualizer
                points = [pt.coord for pt in reconstruction.points3D.values()]
                cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
                info = {"iteration": len(used_images),"points": len(points),"cameras": len(cameras), "error": rms_error,
                        "time": time.time() - start, "pair": next_pair,"active_cams": used_images,"status": "Global BA"}
                stats_log.append(info)
                if vis is not None:
                    vis.post_update(points, cameras, global_flag=True, info=info)


                start = time.time() #Count BA time
                print("Running global bundle adjustment after filtering...")
                res = global_bundle_adjustment(
                    reconstruction,
                    K,
                    optimize_cameras=None,   # None -> all cameras
                    optimize_points=None,    # None -> all points
                    fixed_cameras=None,      # None -> function will fix first camera automatically
                    verbose=2,
                    f_scale=F_SCALE_GLOBAL,
                    xtol=X_TOL,
                    ftol=F_TOL,
                    gtol=G_TOL
                                
                )

                print(f"Running [Strict Reprojection filter] on all points (thresh={STRICT_REPROJ_THRESH}px) ...")
                reconstruction.filter_points_by_reprojection(thresh=STRICT_REPROJ_THRESH)

                # RMS error from residuals
                if hasattr(res, "fun"):
                    rms_error = np.sqrt(np.mean(res.fun**2))
                    residuals = res.fun.reshape(-1, 2)
                    rms_error_per_obs = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
                else:
                    rms_error = -1

                print("Global BA done.")
            
            print(f"==================== Added {len(used_images)} cameras to reconstructon ====================")
            
    print("================================== Running Final Cleanup and global BA =) ==================================")

    points = [pt.coord for pt in reconstruction.points3D.values()]
    cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
    info = {"iteration": len(used_images),"points": len(points),"cameras": len(cameras),"error": rms_error,
            "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Final filtering and global BA"}
    stats_log.append(info)
    if vis is not None:
        vis.post_update(points, cameras, global_flag=True, info=info)

    print(f"Running [Loose Reprojection filter] on all points (thresh={LOOSE_REPROJ_THRESH}px) ...")
    reconstruction.filter_points_by_reprojection(thresh=LOOSE_REPROJ_THRESH)

    print("Running [Spatial Outlier filter] ...")
    reconstruction.filter_points_by_spatial_outliers(k=5, z_thresh=3.0)

    print("Merging close-camera points (keypoint-based)...")
    reconstruction.merge_points_from_close_cameras(match_data, min_matches=MIN_MATCHES)

    print("Running final global bundle adjustment (all cameras + all points)...")
    start = time.time() #Count BA time
    res = global_bundle_adjustment(
        reconstruction,
        K,
        optimize_cameras=None,   # None -> all cameras
        optimize_points=None,    # None -> all points
        fixed_cameras=None,      # None -> function will fix first camera automatically
        verbose=2,
        f_scale=F_SCALE_GLOBAL,
        xtol=X_TOL,
        ftol=F_TOL,
        gtol=G_TOL
    )

    print(f"Running [Strict Reprojection filter] on all points (thresh={STRICT_REPROJ_THRESH}px) ...")
    reconstruction.filter_points_by_reprojection(thresh=STRICT_REPROJ_THRESH)

    print("================================== Global BA done :D ==================================")
    
    # RMS error from residuals
    if hasattr(res, "fun"):
        rms_error = np.sqrt(np.mean(res.fun**2))
        residuals = res.fun.reshape(-1, 2)
        rms_error_per_obs = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
    else:
        rms_error = -1

    print(f"================================== Saving Pointcloud (Coarse) at:  {pcd_output_coarse} ==================================")
    save_reconstruction_points(reconstruction, filename=pcd_output_coarse)
    print(f"================================== Saving cameras (Coarse) at:  {cameras_output_coarse} ==================================")
    save_cameras_np(reconstruction, filename=cameras_output_coarse)

    print(f"================================== Starting fine reconstruction :D ... ==================================")
    

    # Updating visualizer
    points = [pt.coord for pt in reconstruction.points3D.values()]
    cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in reconstruction.cameras.items()]
    info = {"iteration": 0,"points": len(points),"cameras": len(cameras),"error": rms_error,
            "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Fine Reconstruction"}
    stats_log.append(info)
    if vis is not None:
        vis.post_update(points, cameras, info=info)

    fine_reconstruction = Reconstruction()
    #Combine both dictionaries, with the used_pairs
    match_data = used_pairs | match_data
    match_data = dict(sorted(match_data.items(), key=lambda x: len(x[1]), reverse=True)) # Sort it again
    fine_reconstruction.cameras = reconstruction.cameras
    
    cameras = [(idx, cam.R, cam.t, cam.K, imgs_paths[idx]) for idx, cam in fine_reconstruction.cameras.items()]

    it = 0
    pair_keys_matches = list(match_data.keys())
    for pair_key in pair_keys_matches:
        pair_matches = match_data[pair_key]

        # If the matches have les than the min of matches, then it will be terminated 
        # This because is sorted and all the remaining matches will be <200
        if len(pair_matches) < MIN_MATCHES:
            break

        i, j = map(int, pair_key.split("_"))

        cam_i = fine_reconstruction.cameras.get(i, None)
        cam_j = fine_reconstruction.cameras.get(j, None)
        if cam_i is None or cam_j is None:          # One of the cameras is not added, therefore can't be triangulated
            continue

        pair_matches = load_matches(match_data, pair_key)

        kp1, kp2 = kp_data[i], kp_data[j]

        before = len(fine_reconstruction.points3D)
        fine_reconstruction.triangulate_new_points(i, j, kp1, kp2, pair_matches,
                                    min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
        after = len(fine_reconstruction.points3D)

        # Updating visualizer
        points = [pt.coord for pt in fine_reconstruction.points3D.values()] 
        info = {"iteration": it,"points": len(points),"cameras": len(cameras),"error": rms_error,
                "time": time.time() - start,"pair": next_pair,"active_cams": [i, j],"status": "Fine reconstruction"}
        stats_log.append(info)
        if vis is not None:
            vis.post_update(points, cameras, info=info)

        print("Merging close-camera points (keypoint-based)...")
        reconstruction.merge_points_from_close_cameras(match_data, min_matches=MIN_MATCHES)

        it+=1

    # Updating visualizer
    points = [pt.coord for pt in fine_reconstruction.points3D.values()]  
    info = {"iteration": it,"points": len(points),"cameras": len(cameras),"error": rms_error,
            "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Filtering: Strict Reprojection"}
    stats_log.append(info)  
    if vis is not None:
        vis.post_update(points, cameras, info=info)  

    print(f"Running final [Strict Reprojection filter] on all points (thresh={STRICT_REPROJ_THRESH}px) ...")
    reconstruction.filter_points_by_reprojection(thresh=STRICT_REPROJ_THRESH)

    # Updating visualizer
    points = [pt.coord for pt in fine_reconstruction.points3D.values()]  
    info = {"iteration": it,"points": len(points),"cameras": len(cameras),"error": rms_error,
            "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Filtering: Spatial Outliers"}
    stats_log.append(info)
    if vis is not None:
        vis.post_update(points, cameras, info=info)


    print("Running final [Spatial Outlier filter] on all points ...")
    fine_reconstruction.filter_points_by_spatial_outliers(k=5, z_thresh=3.0)

    # Updating visualizer
    points = [pt.coord for pt in fine_reconstruction.points3D.values()]
    info = {"iteration": it,"points": len(points),"cameras": len(cameras),"error": rms_error,
            "time": time.time() - start,"pair": next_pair,"active_cams": used_images,"status": "Merging points/observations"}
    stats_log.append(info)   
    if vis is not None:
        vis.post_update(points, cameras, info=info)

    print("Merging close-camera points (keypoint-based)...")
    reconstruction.merge_points_from_close_cameras(match_data, min_matches=MIN_MATCHES)

    # Updating visualizer
    points = [pt.coord for pt in fine_reconstruction.points3D.values()]
    info = {"iteration": it,"points": len(points),"cameras": len(cameras),"error": rms_error,
            "time": time.time() - start,"pair": next_pair,"active_cams": [],"status" : "Done :D"} 
    stats_log.append(info)
    if vis is not None:
        vis.post_update(points, cameras, info=info)

    print(f"================================== Saving Pointcloud (Fine) at:  {pcd_output_fine} ==================================")
    save_reconstruction_points(fine_reconstruction, filename=pcd_output_fine)
    print(f"================================== Saving cameras (Fine) at:  {cameras_output_fine} ==================================")
    save_cameras_np(fine_reconstruction, filename=cameras_output_fine)
    print(f"================================== Saving stats log at:  {recons_dir}{dataset}_stats.csv ==================================")
    df = pd.DataFrame(stats_log)
    df.to_csv(recons_dir + f"{dataset}_stats.csv", index=False)
    print(f"Stats saved at {recons_dir}{dataset}_stats.csv")

    return reconstruction

if __name__ == "__main__":
    ################### Paths and settings of dataset #####################
    dataset = "dog"
    image_dir = "data/images/"+ dataset

    recons_dir = "data/reconstructions/" + dataset + "/"
    os.makedirs(recons_dir, exist_ok=True)
    data_output_coarse =  dataset + "_coarse"
    data_output_fine = dataset + "_fine"

    imgs_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    imgs_paths = [os.path.join(image_dir, f) for f in imgs_list]

    N_IMGS = len(imgs_list)                     # Number of cams to use for incremental SfM
    #N_IMGS = 40

    feature_file = "data/features/" + dataset + "-geom.npz"

    # pcd_output_coarse = recons_dir + data_output_coarse + "_" + str(N_IMGS) + "_imgs.ply"
    # cameras_output_coarse = recons_dir + data_output_coarse + "_" + str(N_IMGS) + "_imgs.ply"

    # pcd_output_fine = recons_dir + data_output_fine + "_" + str(N_IMGS) + "_imgs.ply"
    # cameras_output_fine = recons_dir + data_output_fine + "_" + str(N_IMGS) + "_imgs.ply"

    pcd_output_coarse = recons_dir + data_output_coarse + ".ply"
    cameras_output_coarse = recons_dir + data_output_coarse

    pcd_output_fine = recons_dir + data_output_fine + ".ply"
    cameras_output_fine = recons_dir + data_output_fine 

    K = build_intrinsic_matrix(imgs_paths[0])

    img = cv2.imread(imgs_paths[0])
    img_size = (img.shape[1], img.shape[0])
    ######################################################################
    ############################ Starting GUI ############################
    gui.Application.instance.initialize()
    vis = SfMVisualizer(img_size=img_size)
    #vis = SfMVisualizer_simple(img_size=img_size)
    #reconstruction = incremental_sfm(N_IMGS, feature_file, K, app=app)
    threading.Thread(target=incremental_sfm, args=(N_IMGS, feature_file, K, vis), daemon=True).start()
    gui.Application.instance.run()