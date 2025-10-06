import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

def create_observations(reconstruction, local_cam_idxs=None, optimize_points=None):
    """
    Create and/or filter observations to include only relevant ones for bundle adjustment.

    Args:
        reconstruction: Reconstruction object
        local_cam_idxs: list of cameras to optimize (None = all)
        optimize_points: list of points to optimize (None = all)
    
    Returns:
        obs_filtered: list of (cam_idx, pt_idx, xy)
        pts_to_opt: dict global_pt_idx -> local_idx
    """
    observations = []
    pts_to_opt = {}
    pts_counter = 0

    # Default: all cameras
    if local_cam_idxs is None:
        local_cam_idxs = list(reconstruction.cameras.keys())
    local_cam_set = set(local_cam_idxs)

    # Default: all points
    if optimize_points is not None:
        optimize_points_set = set(optimize_points)
    else:
        optimize_points_set = None

    for pt_idx, pt3d in reconstruction.points3D.items():
        # Skip points not in optimize_points
        if optimize_points_set is not None and pt_idx not in optimize_points_set:
            continue

        # Only include points observed by any local camera
        cams_observing = set(pt3d.observations.keys())
        if not cams_observing & local_cam_set:
            continue

        # Assign local index if point will be optimized
        if pt_idx not in pts_to_opt:
            pts_to_opt[pt_idx] = pts_counter
            pts_counter += 1

        # Add observations: only those in local cameras or those observing points to optimize
        for cam_idx, kp_idx in pt3d.observations.items():
            cam = reconstruction.cameras[cam_idx]
            xy = cam.kp[kp_idx].pt
            observations.append((cam_idx, pt_idx, np.array(xy)))

    return observations, pts_to_opt

def build_jac_sparsity(n_var_cams, n_var_pts, observations, cam_idx_map, pts_idx_map):
    """
    Returns a sparse matrix (scipy.sparse.csr) of shape
    (2*num_obs, 6*n_var_cams + 3*n_var_pts) indicating which
    residuals depend on which variables.
    """
    n_obs = len(observations)
    n_res = 2 * n_obs
    n_vars = 6 * n_var_cams + 3 * n_var_pts
    S = lil_matrix((n_res, n_vars), dtype=bool)

    for i, (cam_idx, pt_idx, xy) in enumerate(observations):
        row = 2 * i
        # camera part
        cam_local = cam_idx_map.get(cam_idx, None)
        if cam_local is not None:
            cam_base = cam_local * 6
            S[row:row+2, cam_base:cam_base+6] = True
        # point part
        pt_local = pts_idx_map.get(pt_idx, None)
        if pt_local is not None:
            pt_base = 6 * n_var_cams + pt_local * 3
            S[row:row+2, pt_base:pt_base+3] = True

    return S.tocsr()

def _reprojection_error(params, cam_idx_map, pts_idx_map, cameras, points3D, observations, K):
    """
    Residual function for global BA.

    params layout:
        [cam0_rvec(3), cam0_t(3), cam1_rvec(3), cam1_t(3), ..., pts0(3), pts1(3), ...]
    cam_idx_map: dict global_cam_idx -> local_camera_param_index (0..n_var_cams-1)
    pts_idx_map: dict global_pt_idx -> local_point_index (0..n_var_points-1)
    cameras: dict of Camera objects (global indices)
    points3D: list of Point3D objects (global indices)
    observations: list of (cam_idx, pt_idx, xy)
    K: camera intrinsics
    """
    n_cam_vars = len({v for v in cam_idx_map.values()})
    # extract camera params
    cam_params_len = 6 * n_cam_vars
    cam_params = params[:cam_params_len] if cam_params_len > 0 else np.array([])

    pts_flat = params[cam_params_len:].reshape((-1, 3)) if params.size > cam_params_len else np.zeros((0,3))

    residuals = []

    # cache computed rvec/tvec for variable cams to avoid repeated Rodrigues
    rvec_cache = {}
    tvec_cache = {}

    for cam_idx, pt_idx, xy in observations:
        # point local index (if this point is optimized)
        pt_local = pts_idx_map.get(pt_idx, None)

        # get 3D position (from variable params if optimized, otherwise from points3D)
        if pt_local is not None:
            X = pts_flat[pt_local].reshape(1,3)
        else:
            X = points3D[pt_idx].coord.reshape(1,3)

        # get camera pose (variable or fixed)
        cam_local = cam_idx_map.get(cam_idx, None)
        if cam_local is not None:
            # extract rvec,tvec from cam_params
            base = cam_local * 6
            rvec = cam_params[base: base+3]
            tvec = cam_params[base+3: base+6]
        else:
            cam = cameras[cam_idx]
            # cache Rodrigues conversion
            if cam_idx not in rvec_cache:
                rvec_c, _ = cv2.Rodrigues(cam.R)
                rvec_cache[cam_idx] = rvec_c.ravel()
                tvec_cache[cam_idx] = cam.t.ravel()
            rvec = rvec_cache[cam_idx]
            tvec = tvec_cache[cam_idx]

        x_proj, _ = cv2.projectPoints(X, rvec, tvec, K, None)
        err = xy - x_proj.ravel()
        residuals.append(err)

    if len(residuals) == 0:
        return np.array([])

    return np.concatenate(residuals)

def _reprojection_error_vectorized(params, cam_idx_map, pts_idx_map, cameras, points3D, observations, K):
    """
    Vectorized reprojection error for bundle adjustment (robust to single-point cases
    and handles both variable and fixed cameras).
    """
    n_var_cams = len(cam_idx_map)
    cam_params_len = 6 * n_var_cams
    cam_params = params[:cam_params_len] if cam_params_len > 0 else np.array([])
    pts_flat = params[cam_params_len:].reshape((-1, 3)) if params.size > cam_params_len else np.zeros((0,3))

    # Precompute camera R/t for variable cams
    r_cache = {}
    t_cache = {}
    for cam_idx, local_idx in cam_idx_map.items():
        base = local_idx * 6
        rvec = cam_params[base:base+3]
        tvec = cam_params[base+3:base+6]
        R, _ = cv2.Rodrigues(rvec)
        r_cache[cam_idx] = R
        t_cache[cam_idx] = tvec.ravel()

    n_obs = len(observations)
    residuals = np.zeros(2 * n_obs)

    # iterate cameras that actually have observations
    cams_with_obs = sorted(set([obs[0] for obs in observations]))
    for cam_idx in cams_with_obs:
        # Indices of observations for this camera
        obs_idx = np.array([i for i, (cidx, _, _) in enumerate(observations) if cidx == cam_idx], dtype=int)
        obs_points_idx = [observations[i][1] for i in obs_idx]
        xy_obs = np.array([observations[i][2] for i in obs_idx])

        # Build Xs (N,3) robustly (handles single point)
        Xs_list = []
        for pt_idx in obs_points_idx:
            pt_local = pts_idx_map.get(pt_idx, None)
            if pt_local is not None:
                Xs_list.append(pts_flat[pt_local])
            else:
                Xs_list.append(points3D[pt_idx].coord)
        Xs = np.array(Xs_list)
        Xs = Xs.reshape(-1, 3)  # force shape (N,3)

        # Get camera R and t (handle fixed cams not present in cam_idx_map)
        R = r_cache.get(cam_idx)
        t = t_cache.get(cam_idx)
        if R is None or t is None:
            cam = cameras[cam_idx]
            # cam.R might already be a rotation matrix
            if cam.R.shape == (3,3):
                R = cam.R
            else:
                R, _ = cv2.Rodrigues(cam.R)
            t = cam.t.ravel()

        # final sanity asserts (will raise if something odd)
        assert R.shape == (3,3), f"R has wrong shape: {R.shape} for cam {cam_idx}"
        t = np.asarray(t).ravel()
        assert t.shape == (3,), f"t has wrong shape: {t.shape} for cam {cam_idx}"
        assert Xs.ndim == 2 and Xs.shape[1] == 3, f"Xs has wrong shape: {Xs.shape} for cam {cam_idx}"

        # Project: transform to camera coordinates, then perspective divide
        X_cam = (R @ Xs.T).T + t  # (N,3)
        # safe depth: avoid division by zero
        z = X_cam[:, 2]
        eps = 1e-8
        z_safe = np.where(np.abs(z) < eps, eps, z)
        x_proj = X_cam[:, 0] / z_safe
        y_proj = X_cam[:, 1] / z_safe
        u = K[0, 0] * x_proj + K[0, 2]
        v = K[1, 1] * y_proj + K[1, 2]

        # Fill residuals (vectorized assignment)
        residuals[2*obs_idx]   = xy_obs[:, 0] - u
        residuals[2*obs_idx+1] = xy_obs[:, 1] - v

    return residuals

def build_jac_sparsity_fast(n_var_cams, n_var_pts, observations, cam_idx_map, pts_idx_map):
    """
    Fast sparse Jacobian construction using COO format.
    """
    n_obs = len(observations)
    n_res = 2 * n_obs
    n_vars = 6 * n_var_cams + 3 * n_var_pts

    rows, cols = [], []

    for i, (cam_idx, pt_idx, _) in enumerate(observations):
        row_base = 2 * i

        # Camera contribution
        cam_local = cam_idx_map.get(cam_idx, None)
        if cam_local is not None:
            cam_base = cam_local * 6
            for j in range(6):
                rows.append(row_base)
                cols.append(cam_base + j)
                rows.append(row_base + 1)
                cols.append(cam_base + j)

        # Point contribution
        pt_local = pts_idx_map.get(pt_idx, None)
        if pt_local is not None:
            pt_base = 6 * n_var_cams + pt_local * 3
            for j in range(3):
                rows.append(row_base)
                cols.append(pt_base + j)
                rows.append(row_base + 1)
                cols.append(pt_base + j)

    data = np.ones(len(rows), dtype=bool)
    S = coo_matrix((data, (rows, cols)), shape=(n_res, n_vars))
    return S.tocsr()

def local_bundle_adjustment(reconstruction, K, local_cam_idxs, verbose=2, f_scale=100.0, 
                            xtol=1e-7, ftol=1e-7, gtol=1e-7, max_nfev=100):
    """
    Local bundle adjustment on a subset of cameras and the points they observe.

    Args:
        reconstruction: Reconstruction object containing cameras and 3D points
        K: Camera intrinsic matrix
        local_cam_idxs: List of camera indices to optimize
        verbose: Verbosity level
        f_scale: Scale parameter for robust loss (huber); residuals larger than this are down-weighted
        xtol, ftol, gtol: Optimization tolerances
        max_nfev: Maximum number of function evaluations
    """
    # step 1:  Determine points to optimize (observed by any local camera)
    pts_to_opt = {}
    pts_counter = 0
    obs_local = []
    cameras = reconstruction.cameras
    points3D = reconstruction.points3D

    # step 2: Build local camera index map
    # keep the oldest of the local cameras fixed, optimize the rest
    fixed_local = {local_cam_idxs[0]} if len(local_cam_idxs) > 0 else set()
    #fixed_local = set()
    var_cams = [c for c in local_cam_idxs if c not in fixed_local]
    cam_idx_map = {cam_idx: i for i, cam_idx in enumerate(var_cams)}
    
    #cam_idx_map = {cam_idx: i for i, cam_idx in enumerate(local_cam_idxs)}

    # step 3: Build Observations and local point index map
    #pts_idx_map = {pt_idx: i for i, pt_idx in enumerate(pts_to_opt.keys())}
    obs_local, pts_to_opt = create_observations(reconstruction, local_cam_idxs=local_cam_idxs)
    pts_idx_map = {pt_idx: i for i, pt_idx in enumerate(pts_to_opt.keys())}

    if len(pts_to_opt) < 4:
        print("Not enough points for Local BA")
        return

    # step 4: Initial parameter vector
    cam_params_list = []
    for cam_idx in var_cams:  # use local_cam_idx to allow every camera to move (not fixed ones)
        cam = cameras[cam_idx]
        rvec, _ = cv2.Rodrigues(cam.R)
        tvec = cam.t.ravel()
        cam_params_list.append(rvec.ravel())
        cam_params_list.append(tvec.ravel())
    cam_params = np.hstack(cam_params_list)

    pts_init = np.array([points3D[pt_idx].coord for pt_idx in pts_to_opt.keys()])
    x0 = np.hstack((cam_params.ravel(), pts_init.ravel()))

    # step 5: Build Jacobian sparsity
    jac_sparsity = build_jac_sparsity_fast(len(var_cams), len(pts_to_opt), obs_local, cam_idx_map, pts_idx_map)
    
    #########################################################
    ########## Least Squares (Modifiy accordingly) ##########
    res = least_squares(
        _reprojection_error_vectorized,
        x0,
        args=(cam_idx_map, pts_idx_map, cameras, points3D, obs_local, K),
        jac_sparsity=jac_sparsity,
        verbose=verbose,
        method='trf',        # could try 'dogbox' too
        loss='soft_l1',
        f_scale=f_scale,        # bigger f_scale -> less damped steps
        x_scale='jac',
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=max_nfev,
        #diff_step=1e-1       # finite difference step for Jacobian, can help larger updates
    )
    #########################################################
    #########################################################

    # step 7: Unpack optimized cameras
    cam_params_len = 6 * len(var_cams)
    cam_opt = res.x[:cam_params_len] if cam_params_len > 0 else np.array([])
    pts_opt = res.x[cam_params_len:].reshape(-1,3) if res.x.size > cam_params_len else np.zeros((0,3))

    for cam_idx, local_idx in cam_idx_map.items():
        base = local_idx * 6
        rvec_opt = cam_opt[base:base+3]
        tvec_opt = cam_opt[base+3:base+6]
        cam = cameras[cam_idx]
        cam.R, _ = cv2.Rodrigues(rvec_opt)
        cam.t = tvec_opt

    # step 8: Update points
    for pt_global, local_idx in pts_idx_map.items():
        points3D[pt_global].coord = pts_opt[local_idx]
    #print(f"Res: {res}")
    return res


def global_bundle_adjustment(reconstruction, K, optimize_cameras=None, optimize_points=None, 
                             fixed_cameras=None, verbose=2, f_scale=10.0, xtol=1e-4, ftol=1e-4, gtol=1e-6, max_nfev=40):
    """
    Global bundle adjustment on all or a subset of cameras and points.

    Args:
        reconstruction: Reconstruction object containing cameras and 3D points
        K: Camera intrinsic matrix
        optimize_cameras: Optional list of camera indices to optimize (None = all)
        optimize_points: Optional list of point indices to optimize (None = all)
        fixed_cameras: Optional list of cameras to keep fixed (None = first camera fixed)
        verbose: Verbosity level
        f_scale: Scale parameter for robust loss (huber); residuals larger than this are down-weighted
        xtol, ftol, gtol: Optimization tolerances
        max_nfev: Maximum number of function evaluations
    """
    cameras = reconstruction.cameras
    all_cam_idxs = list(cameras.keys())
    points3D = reconstruction.points3D

    # Step 1: Collect all observations (cam_idx, pt_idx, xy)
    observations, _ = create_observations(reconstruction)

    # Step 2: Determine cameras to optimize
    if optimize_cameras is None:
        optimize_cameras = all_cam_idxs.copy()
    else:
        optimize_cameras = list(optimize_cameras)

    # Step 3: Determine fixed cameras
    if fixed_cameras is None:
        fixed_cameras = {all_cam_idxs[0]} if len(all_cam_idxs) > 0 else set()
    else:
        fixed_cameras = set(fixed_cameras)

    # Step 4: Filter variable cameras (those that will actually be optimized)
    var_cams = [c for c in optimize_cameras if c not in fixed_cameras]
    cam_idx_map = {cam_idx: i for i, cam_idx in enumerate(var_cams)}

    # Step 5: Determine points to optimize
    n_points = len(points3D)
    if optimize_points is None:
        optimize_points = list(points3D.keys())
    else:
        optimize_points = list(optimize_points)
    pts_idx_map = {pt_idx: i for i, pt_idx in enumerate(optimize_points)}

    # Step 6: Build initial parameter vector
    cam_params_list = []
    for cam_idx in var_cams:
        cam = cameras[cam_idx]
        rvec, _ = cv2.Rodrigues(cam.R)  # rotation matrix → rotation vector
        tvec = cam.t.ravel()
        cam_params_list.append(rvec.ravel())
        cam_params_list.append(tvec.ravel())
    cam_params = np.hstack(cam_params_list) if len(cam_params_list) > 0 else np.array([])

    # Step 7: Flatten initial 3D points
    pts_init = [points3D[pt_idx].coord.ravel() for pt_idx in optimize_points]
    pts_init = np.array(pts_init).reshape(-1,3) if len(pts_init) > 0 else np.zeros((0,3))

    # Step 8: Concatenate camera + point parameters
    x0 = np.hstack((cam_params.ravel(), pts_init.ravel())) if cam_params.size or pts_init.size else np.zeros(0)
    if x0.size == 0:
        print("No parameters to optimize.")
        return None

    # Step 9: Build sparsity pattern for Jacobian
    jac_sparsity = build_jac_sparsity_fast(len(var_cams), len(optimize_points),
                                      observations, cam_idx_map, pts_idx_map)

    # Step 10: Run least-squares optimization
    
    #########################################################
    ########## Least Squares (Modifiy accordingly) ##########
    res = least_squares(
        _reprojection_error_vectorized,
        x0,
        args=(cam_idx_map, pts_idx_map, cameras, points3D, observations, K),
        jac_sparsity=jac_sparsity,
        verbose=verbose,
        method='trf',        # could try 'dogbox' too
        loss='soft_l1',
        #f_scale=f_scale,        # bigger f_scale -> less damped steps
        x_scale='jac',
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        max_nfev=max_nfev,
        #diff_step=1e-1       # finite difference step for Jacobian, can help larger updates
    )
    #########################################################
    #########################################################

    # Step 11: Unpack optimized camera parameters
    cam_params_len = 6 * len(var_cams)
    cam_opt = res.x[:cam_params_len] if cam_params_len > 0 else np.array([])
    pts_opt = res.x[cam_params_len:].reshape(-1,3) if res.x.size > cam_params_len else np.zeros((0,3))

    # Step 12: Update camera rotations and translations
    for cam_idx, local_idx in cam_idx_map.items():
        base = local_idx * 6
        rvec_opt = cam_opt[base:base+3]
        tvec_opt = cam_opt[base+3:base+6]
        cam = cameras[cam_idx]
        cam.R, _ = cv2.Rodrigues(rvec_opt)
        cam.t = tvec_opt

    # Step 13: Update 3D point coordinates
    for pt_global, local_idx in pts_idx_map.items():
        points3D[pt_global].coord = pts_opt[local_idx]

    return res
