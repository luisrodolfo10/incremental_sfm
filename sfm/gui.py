import open3d as o3d
import numpy as np
import open3d as o3d
import numpy as np
import threading
import time
import cv2
from open3d.visualization import gui, rendering
from sfm.utils import load_cameras_np

def create_camera_frustum_visualizer(extrinsic, K, img_size=(1280, 720), scale=0.5, color=[0,0,1]):
    """
    Create a simple camera frustum as a LineSet for Open3D visualization.

    extrinsic : 4x4 world-to-camera matrix
    K         : 3x3 intrinsic matrix
    width     : image width in pixels
    height    : image height in pixels
    scale     : depth of frustum
    color     : RGB color
    """
    # Convert extrinsic to camera-to-world
    cam2world = np.linalg.inv(extrinsic)
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]

    # Camera center in world coordinates
    C = t
    width = img_size[0]
    height = img_size[1]

    # Image plane corners in pixel coordinates (homogeneous)
    img_corners = np.array([
        [0,      0,       1],   # top-left
        [width,  0,       1],   # top-right
        [width,  height,  1],   # bottom-right
        [0,      height,  1]    # bottom-left
    ])

    # Backproject pixel coords to camera space
    K_inv = np.linalg.inv(K)
    corners_cam = []
    for uv1 in img_corners:
        ray_dir = K_inv @ uv1
        ray_dir /= np.linalg.norm(ray_dir)
        corners_cam.append(ray_dir * scale)
    corners_cam = np.array(corners_cam)

    # Transform corners to world space
    corners_world = (R @ corners_cam.T).T + t

    # Stack points: center + 4 corners
    points = np.vstack((C, corners_world))

    # Define frustum lines
    lines = [
        [0,1],[0,2],[0,3],[0,4],  # center to corners
        [1,2],[2,3],[3,4],[4,1]   # edges of image plane
    ]

    # Build Open3D LineSet
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    frustum.paint_uniform_color(color)

    return frustum, points

class SfMVisualizer:
    GEOM_NAME = "Points"
    CAM_PREFIX = "Cam"
    FRAME_PREFIX = "Frame"

    def __init__(self, window_title="SfM Visualizer", width=1400, height=800, img_size=(1280, 720)):
        self.lock = threading.Lock()
        self.is_done = False
        self.width = width
        self.height = height
        self.img_size = img_size
        self.left_panel_width = int(0.20 * self.width) #  % of the total width

        # hover/selection state
        self.hovered_cam_idx = None
        self.selected_cam_idx = None

        # camera storage: cam_idx -> dict with verts, center, radius, material, geom_name, frame_name
        self.camera_frustums = {}
        self.camera_params = {}  # cam_idx -> (R, t, K)
        self.active_cams = set()

        # small cache of current colors to avoid redundant modify calls
        self._frustum_color_cache = {}

        # UI
        self.app = gui.Application.instance
        self.window = self.app.create_window(window_title, width, height)
        self.window.set_on_close(self.on_close)

        em = self.window.theme.font_size
        separation = int(0.5 * em)

        # --- Scene widget ---
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0, 0, 0, 1])

        # --- Left info panel ---
        self.left_panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self.left_panel.frame = gui.Rect(0, 0, self.left_panel_width, self.height)

        self.value_constraint = gui.Widget.Constraints()
        self.value_constraint.width = self.left_panel_width
        self.value_constraint.height = em * 2

        grid = gui.VGrid(2, spacing = 2)
        self.rows = {}

        for key in ["Iteration", "Points", "Cameras", "Reproj Error", 
                    "Last Iter Time", "Last Pair", "Bundle Cams", 
                    "Elapsed", "Selected cam", "Filename", "Status"]:
            
            key_label = gui.Label(key + ": ")
            key_label.text_color = gui.Color(0.7, 0.7, 0.7)

            value_label = gui.Label("--------------------------------")
            value_label.text_color = gui.Color(1,1,1)
            
            # Set preferred width for value
            value_width = int(self.left_panel_width * 0.4)
            value_label.frame = gui.Rect(0, 0, value_width, em*2)

            # Add key and value to the grid
            grid.add_child(key_label)
            grid.add_child(value_label)

            self.rows[key] = value_label

        self.rows["Bundle Cams"].text_color = gui.Color(0, 1, 0) # Bundle cams show in green

        self.left_panel.add_child(grid)
        
        self.start_time = time.time()

        self.done_flag = False

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.left_panel)

        # --- Point cloud (initially empty) ---
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0
        self.scene_widget.scene.add_geometry(self.GEOM_NAME, self.pcd, self.mat)

        # default line material (used as base; individual materials per frustum are stored)
        self._default_line_mat = rendering.MaterialRecord()
        self._default_line_mat.shader = "unlitLine"
        self._default_line_mat.line_width = 1.0

        # Setup camera so scene is visible
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self.scene_widget.setup_camera(30, bbox, [0, 0, 0])
        # invert Y-axis 
        self.scene_widget.look_at(
            np.array([[0], [0], [0]], dtype=np.float32),
            np.array([[0], [0], [20]], dtype=np.float32),
            np.array([[0], [-1], [0]], dtype=np.float32)
        )

        # floating hover label
        self.hover_label = gui.Label("")
        self.hover_label.visible = False
        self.window.add_child(self.hover_label)

        # throttle hover updates
        self.last_hover_update = 0.0
        #self.hover_min_interval = 0.04  # seconds (25 Hz)
        self.hover_min_interval = 0.06  # seconds Reduce the freqency of the ray_checks for selecting cam

        # timer thread
        threading.Thread(target=self._run_timer, daemon=True).start()

        # Image viewers
        self.heatmap_checkbox = gui.Checkbox("Heatmap Overlay")
        self.heatmap_checkbox.checked = True  # default ON
        self.heatmap_checkbox.set_on_checked(self._on_heatmap_toggled)
        self.left_panel.add_child(self.heatmap_checkbox)
        self.left_panel.add_fixed(15) # 15 pixels down

        # --- Right panel (for image widgets) ---
        self.right_panel_width = int(0.20 * self.width)  # 25% of window width
        self.right_panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self.right_panel.frame = gui.Rect(self.width - self.right_panel_width, 0,
                                        self.right_panel_width, self.height)

        # Image widgets
        self.cam_original_view = gui.ImageWidget()
        self.cam_projected_view = gui.ImageWidget()
        self.cam_overlay_view = gui.ImageWidget()

        # Fixed size for images
        fixed_width, fixed_height = 320, 240
        self.cam_original_view.frame = gui.Rect(0, 0, fixed_width, fixed_height)
        self.cam_projected_view.frame = gui.Rect(0, 0, fixed_width, fixed_height)
        self.cam_overlay_view.frame = gui.Rect(0, 0, fixed_width, fixed_height)

        # Add labels + images to right panel
        self.left_panel.add_child(gui.Label("Original Image"))
        self.left_panel.add_child(self.cam_original_view)
        self.right_panel.add_child(gui.Label("Camera Projection of Points"))
        self.right_panel.add_child(self.cam_projected_view)
        self.right_panel.add_fixed(10) #10 pixels
        self.right_panel.add_child(gui.Label("Overlay Points"))
        self.right_panel.add_child(self.cam_overlay_view)


        # Add right panel to window
        self.window.add_child(self.right_panel)

        # Add buttons save and re-center

        reset_button = gui.Button("Reset View")
        save_projected_button = gui.Button("Save Projected Points Image")
        save_overlay_button = gui.Button("Save Overlay Image")
        reset_button.set_on_clicked(self._on_reset_view)
        save_projected_button.set_on_clicked(self._on_save_projected)
        save_overlay_button.set_on_clicked(self._on_save_overlay)
        
        self.left_panel.add_fixed(15)
        self.left_panel.add_child(reset_button)
        self.right_panel.add_fixed(15)
        self.right_panel.add_child(save_projected_button)
        self.right_panel.add_fixed(10)
        self.right_panel.add_child(save_overlay_button)
        self.right_panel.add_fixed(10)
        self.saved_label = gui.Label(" "*20)
        self.right_panel.add_child(self.saved_label)
        
        self.right_panel.add_stretch()  # some spacing
        author_label = gui.Label("© Luis Rodolfo Macias 2025")
        author_label.text_color = gui.Color(0.7, 0.7, 0.7)  # subtle gray

        font_monospace = gui.FontDescription(
            typeface="monospace", style=gui.FontStyle.BOLD, point_size=14)

        font_id_monospace = gui.Application.instance.add_font(font_monospace)
        author_label.font_id = font_id_monospace  
        self.right_panel.add_child(author_label)

        self.projected_image = None
        self.overlay_image = None
        
        # # hook mouse
        self._setup_interaction()

    # ---------------------------
    # layout & window callbacks
    # ---------------------------
    def _on_layout(self, layout_context):
        left_width = self.left_panel_width
        right_width = self.right_panel_width
        center_width = self.width - left_width - right_width

        # Position left panel
        self.left_panel.frame = gui.Rect(0, 0, left_width, self.height)

        # Position 3D scene in the center
        self.scene_widget.frame = gui.Rect(left_width, 0, center_width, self.height)

        # Position right panel
        self.right_panel.frame = gui.Rect(left_width + center_width, 0, right_width, self.height)

    def on_close(self):
        with self.lock:
            self.is_done = True
        return True

    def _update_timer(self, elapsed):
        self.rows["Elapsed"].text = f"{elapsed:.1f}"

    def _run_timer(self):
        while not self.is_done:
            elapsed = time.time() - self.start_time
            # Post update to the GUI thread
            if not self.done_flag:
                try:
                    self.app.post_to_main_thread(self.window, lambda e=elapsed: self._update_timer(e))
                except Exception:
                    pass
                time.sleep(0.1)
            else:
                return

    # ---------------------------
    # camera / geometry management
    # ---------------------------
    def _init_camera_geom(self, cam_idx, R, t, K, image_path="", active=False):
        """Create frustum geometry, material, compute bounding sphere and add to scene."""
        name = f"{self.CAM_PREFIX}_{cam_idx}"
        frame_name = f"{self.FRAME_PREFIX}_{cam_idx}"

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t.reshape(3)

        # create frustum geometry + verts (center + 4 corners)
        # material per frustum so we can change color cheaply
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 1.0
        initial_color = [0, 1, 0] if active else [1, 0, 0]  # green if active else red
        #mat.base_color = initial_color

        frustum_geom, verts = create_camera_frustum_visualizer(extrinsic, K, self.img_size, scale=1, color=initial_color)

        # add frustum & coordinate frame
        if self.scene_widget.scene.has_geometry(name):
            self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(name, frustum_geom, self.line_mat)

        # coord frame mesh
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame.transform(np.linalg.inv(extrinsic))
        if self.scene_widget.scene.has_geometry(frame_name):
            self.scene_widget.scene.remove_geometry(frame_name)
        self.scene_widget.scene.add_geometry(frame_name, frame, self.mat)

        # bounding sphere (coarse)
        center = verts.mean(axis=0)
        radius = np.linalg.norm(verts - center, axis=1).max()

        # store
        self.camera_frustums[cam_idx] = {
            "verts": verts,
            "center": center,
            "radius": radius,
            "name": name,
            "frame_name": frame_name,
            "geom": frustum_geom
        }
        self.camera_params[cam_idx] = (R.copy(), t.copy(), K.copy(), image_path)
        self._frustum_color_cache[cam_idx] = initial_color

    def _remove_camera_geom(self, cam_idx):
        d = self.camera_frustums.get(cam_idx)
        if not d:
            return
        if self.scene_widget.scene.has_geometry(d["name"]):
            self.scene_widget.scene.remove_geometry(d["name"])
        if self.scene_widget.scene.has_geometry(d["frame_name"]):
            self.scene_widget.scene.remove_geometry(d["frame_name"])
        self.camera_frustums.pop(cam_idx, None)
        self.camera_params.pop(cam_idx, None)
        self._frustum_color_cache.pop(cam_idx, None)

    def _update_frustum_colors(self):
        """Update frustum colors but only when they actually change (cheap)."""
        for cam_idx, d in self.camera_frustums.items():
            name = d["name"]

            if cam_idx == self.selected_cam_idx:
                color = [0, 0, 1] # blue
            elif cam_idx == self.hovered_cam_idx:
                color = [1, 1, 0] # yellow
            elif cam_idx in self.active_cams:
                color = [0, 1, 0] # green
            else:
                color = [1, 0, 0]  # red

            # # Only update if changed
            if self._frustum_color_cache.get(cam_idx) != color:
               self.scene_widget.scene.remove_geometry(name)
               lineset = self.camera_frustums[cam_idx]["geom"]
               lineset.paint_uniform_color(color)
               self.scene_widget.scene.add_geometry(name, lineset, self.line_mat)
               self._frustum_color_cache[cam_idx] = color

    # ---------------------------
    # high-performance ray-hit test
    # ---------------------------
    def _camera_under_ray(self, origin, direction, max_vertex_dist=0.5):
        """
        Return cam_idx under ray or None.
        - origin, direction must be numpy arrays; direction should be normalized.
        - We first do a bounding sphere reject, then a vertex distance check.
        """
        best_idx = None
        best_dist = float("inf")

        for cam_idx, d in self.camera_frustums.items():
            center = d["center"]
            radius = d["radius"]

            # project center onto ray (scalar)
            to_center = center - origin
            proj = np.dot(to_center, direction)

            # quick reject: if projected closest approach is far from center by more than radius + margin
            closest_point = origin + proj * direction
            if np.linalg.norm(closest_point - center) > (radius + max_vertex_dist):
                continue

            # fine check: distance to frustum vertices (distance to infinite line)
            verts = d["verts"]
            # direction should be normalized -> no division needed
            dists = np.linalg.norm(np.cross(verts - origin, direction), axis=1)
            min_dist = dists.min()
            if min_dist < best_dist:
                best_dist = min_dist
                best_idx = cam_idx

        if best_idx is not None and best_dist <= max_vertex_dist:
            return best_idx
        return None

    # ---------------------------
    # interaction / mouse
    # ---------------------------
    def _setup_interaction(self):
        self.scene_widget.set_on_mouse(self._on_mouse_event)

    def _coords_to_widget(self, event):
        """Convert absolute event coords to widget-local coords (pixels from top-left of widget)."""
        mx = event.x - self.scene_widget.frame.x
        my = event.y - self.scene_widget.frame.y
        return mx, my

    def _unproject_ray_from_event(self, event):
        """Return (origin, dir) in world coords for current mouse event (widget-space coords)."""
        mx_scene, my_scene = self._coords_to_widget(event)
        view_w = self.scene_widget.frame.width
        view_h = self.scene_widget.frame.height

        cam = self.scene_widget.scene.camera
        near = cam.unproject(mx_scene, my_scene, 0.0, view_w, view_h)
        far  = cam.unproject(mx_scene, my_scene, 0.99, view_w, view_h)
        origin = np.array(near, dtype=float)
        dirv = np.array(far, dtype=float) - origin
        nrm = np.linalg.norm(dirv)
        if nrm < 1e-8:
            return origin, None
        return origin, dirv / nrm

    def _on_mouse_event(self, event):
        # always accept the event
        if event.type == gui.MouseEvent.Type.MOVE:
            # throttle heavy work
            now = time.time()
            if now - self.last_hover_update < self.hover_min_interval:
                return gui.Widget.EventCallbackResult.HANDLED
            self.last_hover_update = now

            origin, direction = self._unproject_ray_from_event(event)
            if direction is None:
                return gui.Widget.EventCallbackResult.HANDLED

            cam_idx = self._camera_under_ray(origin, direction)
            if cam_idx != self.hovered_cam_idx:
                self.hovered_cam_idx = cam_idx
                self._update_frustum_colors()

            # update hover label position + text
            if cam_idx is not None:
                self.hover_label.text = f"Cam {cam_idx}"
                self.hover_label.frame = gui.Rect(event.x + 10, event.y + 10, 120, 20)
                self.hover_label.visible = True
            else:
                self.hover_label.visible = False

            return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            # selection (single-select). Change to check event.button for left/right, or modifiers
            origin, direction = self._unproject_ray_from_event(event)
            if direction is None:
                return gui.Widget.EventCallbackResult.HANDLED

            cam_idx = self._camera_under_ray(origin, direction)
            if cam_idx is not None:
                self.selected_cam_idx = cam_idx
                self.rows["Selected cam"].text = str(cam_idx)
                self.rows["Filename"].text = self.camera_params[cam_idx][3].split("/")[-1] #Getting the last split of the image_path
                self._update_frustum_colors()
                self._on_camera_selected(cam_idx)
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED
    
    def _on_camera_selected(self, cam_idx):
        # Load original image
        img_path = self.camera_params[cam_idx][3]
        img_cv2 = None
        if img_path is not None:
            #img = o3d.io.read_image(img_path)
            img_cv2 = cv2.imread(img_path)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img = o3d.geometry.Image(img_cv2)
            self.cam_original_view.update_image(img)

        # Render from the camera's perspective
        R, t, K = self.camera_params[cam_idx][:3]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t.reshape(3)

        # Use Open3D offscreen renderer
        #  Just the opints
        width, height = self.img_size
        self.projected_image = self._render_overlay_points(extrinsic, K, width, height, heatmap=self.heatmap_checkbox.checked, point_radius=6)
        self.cam_projected_view.update_image(self.projected_image)

        # Overlay points on an image
        self.overlay_image = self._render_overlay_points(extrinsic, K, width, height, background=img_cv2, heatmap=self.heatmap_checkbox.checked, point_radius=6)
        self.cam_overlay_view.update_image(self.overlay_image)


    def _on_heatmap_toggled(self, is_checked):
        """
        Callback when the heatmap checkbox is toggled.
        Re-renders the overlay points with the new heatmap setting.
        """
        if self.selected_cam_idx is None:
            return
        else:
            self._on_camera_selected(self.selected_cam_idx)

    def _render_overlay_points(self, extrinsic, K, width, height, background=None, heatmap=False, point_radius=3):
        """
        Project 3D points into camera view and draw them efficiently.
        Uses precomputed circular masks for variable radii.
        """
        pts = np.asarray(self.pcd.points)

        # Transform to camera coordinates
        pts_cam = (extrinsic[:3, :3] @ pts.T + extrinsic[:3, 3:4]).T

        # Keep points in front of the camera
        mask = pts_cam[:, 2] > 0
        pts_cam = pts_cam[mask]

        if pts_cam.shape[0] == 0:
            overlay_img = np.zeros((height, width, 3), dtype=np.uint8) if background is None else background.copy()
            return o3d.geometry.Image(overlay_img)

        # Depth normalization for heatmap
        if heatmap:
            z = pts_cam[:, 2]
            z_min, z_max = z.min(), z.max()
            if z_max - z_min < 1e-5:
                z_max = z_min + 1e-5
            norm_depth = (z - z_min) / (z_max - z_min)
        else:
            norm_depth = None

        # Project to 2D
        pts_proj = (K @ pts_cam.T).T
        pts_proj[:, :2] /= pts_proj[:, 2:3]

        # Use background image or blank canvas
        overlay_img = background.copy() if background is not None else np.zeros((height, width, 3), dtype=np.uint8)

        # Compute colors + radii vectorized
        if heatmap:
            d = norm_depth
            b = (255 * (1 - np.clip(d * 4 - 1, 0, 1))).astype(np.uint8) # Blue close ponits
            g = (255 * (1 - np.abs(d * 4 - 2))).astype(np.uint8) # Green mid points
            r = (255 * (np.clip(d * 4 - 3, 0, 1))).astype(np.uint8) # Red'er futher points
            # colors = np.stack([b, g, r], axis=1)  # BGR
            colors = np.stack([r, g, b], axis=1)  # RGB
            radii = (point_radius * (1 - d) + point_radius * 0.5 * d).round().astype(np.int32)
        else:
            colors = np.full((pts_proj.shape[0], 3), 255, dtype=np.uint8)
            radii = np.full(pts_proj.shape[0], point_radius, dtype=np.int32)

        # Round and filter valid pixel coords
        coords = pts_proj[:, :2].round().astype(int)
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < width) & \
                (coords[:, 1] >= 0) & (coords[:, 1] < height)
        coords, colors, radii = coords[valid], colors[valid], radii[valid]

        # --- Precompute circular masks for all unique radii ---
        def make_disk(radius):
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            return (x**2 + y**2) <= radius**2

        unique_radii = np.unique(radii)
        disk_kernels = {r: make_disk(r) for r in unique_radii}

        # --- Stamp points with precomputed kernels ---
        for (x, y), color, r in zip(coords, colors, radii):
            kernel = disk_kernels[r]
            kh, kw = kernel.shape

            # Bounding box in image
            x0, x1 = max(0, x-r), min(width, x+r+1)
            y0, y1 = max(0, y-r), min(height, y+r+1)

            # Bounding box in kernel
            kx0 = max(0, r-x)
            ky0 = max(0, r-y)
            kx1 = kx0 + (x1 - x0)
            ky1 = ky0 + (y1 - y0)

            # Stamp circular mask
            overlay_img[y0:y1, x0:x1][kernel[ky0:ky1, kx0:kx1]] = color

        return o3d.geometry.Image(overlay_img)
    
    def _on_reset_view(self):
        if len(self.pcd.points) == 0:
            return
        bbox = self.pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.scene_widget.setup_camera(30, bbox, center)
        # mirar hacia Z positivo, eje Y hacia abajo
        self.scene_widget.look_at(center, center + np.array([0, 0, 20]), [0, -1, 0])

    def _on_save_overlay(self):
        if self.overlay_image is None:
            print("No overlay image to save.")
            return
        # Save with timestamp
        filename = f"overlay_cam_{self.selected_cam_idx}_{int(time.time())}.png"
        img = self.overlay_image
        img_data = np.asarray(img)  # convertir a numpy
        cv2.imwrite(filename, cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR))
        #print(f"Overlay saved to {filename}")
        self.saved_label.text = f"Overlay saved to {filename}"

    def _on_save_projected(self):
        if self.projected_image is None:
            print("No projected image to save.")
            return
        # Save with timestamp
        filename = f"projected_cam_{self.selected_cam_idx}_{int(time.time())}.png"
        img = self.projected_image
        img_data = np.asarray(img)  # convertir a numpy
        cv2.imwrite(filename, cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR))
        #print(f"Projected saved to {filename}")
        self.saved_label.text = f"Projected saved to {filename}"


# debug: draw single ray (keeps only last)
    def _add_ray_debug(self, origin, direction, length=5.0):
        points = [origin, origin + direction * length]
        lines = [[0, 1]]
        colors = [[1.0, 1.0, 0.0]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        name = "debug_ray"
        if self.scene_widget.scene.has_geometry(name):
            self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(name, line_set, self._default_line_mat)
    
    # ---------------------------
    # public update (called from other thread)
    # ---------------------------
    def post_update(self, points, cameras, global_flag=False, info=None):
        """Thread-safe post to GUI thread."""
        self.app.post_to_main_thread(
            self.window,
            lambda: self.update(points, cameras, global_flag=global_flag, info=info)
        )

    def update(self, points, cameras, global_flag=False, info=None):
        """
        points: Nx3 numpy array (or list)
        cameras: iterable of (cam_idx, R, t, K)
        info: optional dict (contains 'active_cams', 'iteration', etc.)
        """
        with self.lock:
            # --- points: only update if size changed or point positions changed ---
            pts_np = np.asarray(points)
            if pts_np.shape[0] != len(self.pcd.points):
                # update pcd
                colors = np.tile([0.7, 0.7, 0.7], (len(pts_np), 1))
                if global_flag:
                    colors[:] = [0, 1, 0]
                elif pts_np.shape[0] > len(self.pcd.points):
                    # new points green
                    colors[len(self.pcd.points):] = [0, 1, 0]

                self.pcd.points = o3d.utility.Vector3dVector(pts_np)
                self.pcd.points = o3d.utility.Vector3dVector(pts_np)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                if self.scene_widget.scene.has_geometry(self.GEOM_NAME):
                    self.scene_widget.scene.remove_geometry(self.GEOM_NAME)
                self.scene_widget.scene.add_geometry(self.GEOM_NAME, self.pcd, self.mat)

            # --- cameras: add new / update removed ---
            incoming_ids = set()
            for cam_idx, R, t, K, image_path in cameras:
                incoming_ids.add(cam_idx)
                # if new camera -> create geometry
                if cam_idx not in self.camera_frustums:
                    self._init_camera_geom(cam_idx, R, t, K, image_path=image_path, active=(cam_idx in (info.get("active_cams", []) if isinstance(info, dict) else [])))
                else:
                    # if parameters changed, we recreate (rare). This keeps things simple:
                    old = self.camera_params.get(cam_idx)
                    if old is None or not np.allclose(old[0], R) or not np.allclose(old[1], t) or not np.allclose(old[2], K):
                        # remove + recreate
                        self._remove_camera_geom(cam_idx)
                        self._init_camera_geom(cam_idx, R, t, K, image_path=image_path, active=(cam_idx in (info.get("active_cams", []) if isinstance(info, dict) else [])))

            # remove cameras that are no longer present
            existing_ids = set(self.camera_frustums.keys())
            for missing in existing_ids - incoming_ids:
                self._remove_camera_geom(missing)
                
            # # --- Updating values ---
            if isinstance(info, dict):
                active = set(info.get("active_cams", []))
                self.active_cams = active
                self._update_frustum_colors()

                if not self.done_flag and "done" in  info["status"].lower():
                    self.done_flag = True

                # Mapping info keys to the grid labels
                info_to_key = {
                    "iteration": "Iteration",
                    "points": "Points",
                    "cameras": "Cameras",
                    "error": "Reproj Error",
                    "time": "Last Iter Time",
                    "pair": "Last Pair",
                    "active_cams": "Bundle Cams",
                    "elapsed": "Elapsed",
                    "selected_cam": "Selected cam",
                    "filename": "Filename",
                    "status": "Status"
                }

                # Update all value labels
                for info_key, grid_key in info_to_key.items():
                    if info_key in info:
                        if info_key == "active_cams":
                            # Special case: show list or "All cams"
                            if len(self.active_cams) < len(self.camera_params):
                                self.rows[grid_key].text = str(info[info_key])
                            else:
                                self.rows[grid_key].text = "All cams"
                        elif info_key == "error":
                            self.rows[grid_key].text = f"{info[info_key]:.4f}"
                        elif info_key == "time":
                            self.rows[grid_key].text = f"{info[info_key]:.3f} s"
                        else:
                            self.rows[grid_key].text = str(info[info_key])

class SfMVisualizer_simple:
    GEOM_NAME = "Points"
    CAM_PREFIX = "Cam"

    def __init__(self, window_title="SfM Visualizer simple", width=1280, height=720, img_size=(1280, 720)):
        self.lock = threading.Lock()
        self.is_done = False
        self.img_size = img_size

        # GUI window
        self.app = gui.Application.instance
        self.window = self.app.create_window(window_title, width, height)
        self.window.set_on_close(self.on_close)

        # Scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0, 0, 0, 1])
        self.window.add_child(self.scene_widget)

        # Dummy initial point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0
        self.scene_widget.scene.add_geometry(self.GEOM_NAME, self.pcd, self.mat)

        # Setup camera to see the scene
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self.scene_widget.setup_camera(30, bbox, [0, 0, 0])
        # invert Y-axis 
        self.scene_widget.look_at(
            np.array([[0], [0], [0]], dtype=np.float32),
            np.array([[0], [0], [20]], dtype=np.float32),
            np.array([[0], [-1], [0]], dtype=np.float32)
        )


        # Line material
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 1.0

        # Track points for "new point" coloring
        self.prev_num_points = 0

    def update(self, points, cameras, info="", global_flag=False):
        """Update the point cloud and cameras in the scene."""
        with self.lock:
            # Update points
            # --- Update point cloud with coloring for "new" points ---
            if len(points) > 0:
                pts_np = np.asarray(points)
                colors = np.tile([0.7, 0.7, 0.7], (len(pts_np), 1))  # old points gray
                if global_flag:
                    colors[:] = [0, 1, 0]
                elif len(pts_np) > self.prev_num_points or len(pts_np):
                    colors[self.prev_num_points:] = [0, 1, 0]  # new points green

                self.prev_num_points = len(points)

                self.pcd.points = o3d.utility.Vector3dVector(pts_np)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)

                if self.scene_widget.scene.has_geometry(self.GEOM_NAME):
                    self.scene_widget.scene.remove_geometry(self.GEOM_NAME)
                self.scene_widget.scene.add_geometry(self.GEOM_NAME, self.pcd, self.mat)

            active_cams = info["active_cams"]

            # Update cameras
            for cam_idx, R, t, K, _ in cameras: #image_path ignored
                name = f"{self.CAM_PREFIX}_{cam_idx}"
                if self.scene_widget.scene.has_geometry(name):
                    self.scene_widget.scene.remove_geometry(name)

                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t.reshape(3)

                # create frustum geometry + verts (center + 4 corners)
                # material per frustum so we can change color cheaply

                color = [0, 1, 0] if cam_idx in active_cams else [1, 0, 0]  # green if active else red
                #mat.base_color = initial_color

                frustum_geom, _ = create_camera_frustum_visualizer(extrinsic, K, self.img_size, scale=1, color=color)

                self.scene_widget.scene.add_geometry(name, frustum_geom, self.line_mat)

    def post_update(self, points, cameras, info="", global_flag=False):
        """Post update safely from another thread."""
        self.app.post_to_main_thread(
            self.window, 
            lambda: self.update(points, cameras, info=info, global_flag=global_flag)
        )

    def on_close(self):
        """Handle window close."""
        with self.lock:
            self.is_done = True
        return True
    
class SimpleReconstrictionViewer:
    def __init__(self, pcd_file, cameras_file, img_size=(1280, 720)):
        self.pcd_file = pcd_file
        self.cameras_file = cameras_file
        self.img_size = img_size

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(
            "Simple Reconstruction viewer", 1280, 720)

        # Scene widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.0

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0

        self.load_reconstruction()

    def load_reconstruction(self):
        # Load point cloud
        pcd = o3d.io.read_point_cloud(self.pcd_file)
        self.scene.scene.add_geometry("PointCloud", pcd, self.mat)

        # Load cameras
        cam_data = load_cameras_np(self.cameras_file)
        for cam_idx, cam in cam_data.items():
            K, R, t = cam['K'], cam['R'], cam['t']
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t.reshape(3)
            geom, _ = create_camera_frustum_visualizer(extrinsic, K, img_size=self.img_size, scale=1.2)
            self.scene.scene.add_geometry(f"Cam_{cam_idx}_{cam_idx}", geom, self.line_mat)

        # Fit the camera to view
        bounds = self.scene.scene.bounding_box
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def run(self):
        self.app.run()

class ReconstructionViewer:
    def __init__(self, pcd_file, cameras_file, width=1280, height=720, img_size=(1280, 720), imgs_paths=[]):
        self.pcd_file = pcd_file
        self.cameras_file = cameras_file
        self.img_size = img_size
        self.imgs_paths = imgs_paths
        self.height = height
        self.width = width

        # GUI
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Reconstruction Viewer", 1280, 720)
        #self.window.set_on_close(self.on_close)

        # Scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0, 0, 0, 1])
        #self.window.add_child(self.scene_widget)

        # Left panel
        em = self.window.theme.font_size
        self.left_panel_width = int(0.2 * 1280)
        self.left_panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self.left_panel.frame = gui.Rect(0, 0, self.left_panel_width, 720)
        
        #self.window.add_child(self.left_panel)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.left_panel)

        # Info labels
        self.rows = {}
        for key in ["Selected cam", "Filename"]:
            key_label = gui.Label(f"{key}: ")
            key_label.text_color = gui.Color(0.7, 0.7, 0.7)
            value_label = gui.Label("--------------------------------")
            value_label.text_color = gui.Color(1, 1, 1)
            self.left_panel.add_child(key_label)
            self.left_panel.add_child(value_label)
            self.rows[key] = value_label

        # floating hover label
        self.hover_label = gui.Label("")
        self.hover_label.visible = False
        self.window.add_child(self.hover_label)


        # throttle hover updates
        self.last_hover_update = 0.0
        #self.hover_min_interval = 0.04  # seconds (25 Hz)
        self.hover_min_interval = 0.06  # seconds Reduce the freqency of the ray_checks for selecting cam

        # Original image viewer
        self.cam_original_view = gui.ImageWidget()
        self.left_panel.add_child(gui.Label("Original Image"))
        self.left_panel.add_child(self.cam_original_view)
        
        # Add reset view button
        reset_button = gui.Button("Reset View")
        reset_button.set_on_clicked(self._on_reset_view)

        # Text
        self.left_panel.add_fixed(15)
        self.left_panel.add_child(reset_button)

        self.left_panel.add_stretch()  # some spacing
        author_label = gui.Label("© Luis Rodolfo Macias 2025")
        author_label.text_color = gui.Color(0.7, 0.7, 0.7)  # subtle gray

        font_monospace = gui.FontDescription(
            typeface="monospace", style=gui.FontStyle.BOLD, point_size=14)

        font_id_monospace = gui.Application.instance.add_font(font_monospace)
        author_label.font_id = font_id_monospace  
        self.left_panel.add_child(author_label)

        # State
        self.camera_frustums = {}
        self.hovered_cam_idx = None
        self.selected_cam_idx = None

        # Materials
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.0

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0

        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(30, bounds, bounds.get_center())

        # Load data
        self.pcf = None
        self.load_reconstruction()

        # Interaction
        self._setup_interaction()

        # Fit camera to view

    def _on_layout(self, layout_context):
        left_width = self.left_panel_width
        # right_width = self.right_panel_width
        # center_width = self.width - left_width - right_width

        # Position left panel
        self.left_panel.frame = gui.Rect(0, 0, left_width, self.height)

        # Position 3D scene in the center
        self.scene_widget.frame = gui.Rect(left_width, 0, self.width, self.height)

    def _on_reset_view(self):
        if len(self.pcd.points) == 0:
            return
        bbox = self.pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.scene_widget.setup_camera(30, bbox, center)
        # mirar hacia Z positivo, eje Y hacia abajo
        self.scene_widget.look_at(center, center + np.array([0, 0, 20]), [0, -1, 0])

    def load_reconstruction(self):
        # Load point cloud
        self.pcd = o3d.io.read_point_cloud(self.pcd_file)
        self.scene_widget.scene.add_geometry("PointCloud", self.pcd, self.mat)
        #print(len(self.pcd.points))
        # Load cameras
        cam_data = load_cameras_np(self.cameras_file)
        for cam_idx_str, cam in cam_data.items():
            cam_idx = int(cam_idx_str)
            name = f"Cam_{cam_idx}"
            K, R, t = cam['K'], cam['R'], cam['t']
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t.reshape(3)
            geom, verts = create_camera_frustum_visualizer(extrinsic, K, img_size=self.img_size, scale=1.0, color=[1, 0, 0])
            self.scene_widget.scene.add_geometry(name, geom, self.line_mat)
            center = verts.mean(axis=0)
            radius = np.linalg.norm(verts - center, axis=1).max()
            self.camera_frustums[cam_idx] = {"name": name, "geom": geom, "center": center, "radius": radius, "verts": verts}
    # Fit the camera to view
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

    def _setup_interaction(self):
        self.scene_widget.set_on_mouse(self._on_mouse_event)

    def _on_mouse_event(self, event):
        # always accept the event
        if event.type == gui.MouseEvent.Type.MOVE:
            # throttle heavy work
            now = time.time()
            if now - self.last_hover_update < self.hover_min_interval:
                return gui.Widget.EventCallbackResult.HANDLED
            self.last_hover_update = now

            origin, direction = self._unproject_ray_from_event(event)
            if direction is None:
                return gui.Widget.EventCallbackResult.HANDLED

            cam_idx = self._camera_under_ray(origin, direction)
            if cam_idx != self.hovered_cam_idx:
                self.hovered_cam_idx = cam_idx
                self._update_frustum_colors()

            # update hover label position + text
            if cam_idx is not None:
                self.hover_label.text = f"Cam {cam_idx}"
                self.hover_label.frame = gui.Rect(event.x + 10, event.y + 10, 120, 20)
                self.hover_label.visible = True
            else:
                self.hover_label.visible = False

            return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            # selection (single-select). Change to check event.button for left/right, or modifiers
            origin, direction = self._unproject_ray_from_event(event)
            if direction is None:
                return gui.Widget.EventCallbackResult.HANDLED

            cam_idx = self._camera_under_ray(origin, direction)
            if cam_idx is not None:
                self.selected_cam_idx = cam_idx
                self.rows["Selected cam"].text = str(cam_idx)
                self.rows["Filename"].text = self.imgs_paths[cam_idx].split("/")[-1] #Getting the last split of the image_path
                self._update_frustum_colors()
                self._on_camera_selected(cam_idx)
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _update_frustum_colors(self):
        """Update frustum colors but only when they actually change (cheap)."""
        for cam_idx, d in self.camera_frustums.items():
            name = d["name"]

            if cam_idx == self.selected_cam_idx:
                color = [0, 0, 1] # blue
            elif cam_idx == self.hovered_cam_idx:
                color = [1, 1, 0] # yellow
            else:
                color = [1, 0, 0] # Red normal camera

            self.scene_widget.scene.remove_geometry(name)
            lineset = self.camera_frustums[cam_idx]["geom"]
            lineset.paint_uniform_color(color)
            self.scene_widget.scene.add_geometry(name, lineset, self.line_mat)

    def _load_original_image(self, cam_idx):
        img_path = self.imgs_path[cam_idx]
        if img_path:
            img_cv2 = cv2.imread(img_path)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img = o3d.geometry.Image(img_cv2)
            self.cam_original_view.update_image(img)

    def _unproject_ray_from_event(self, event):
        """Return (origin, dir) in world coords for current mouse event (widget-space coords)."""
        mx_scene, my_scene = self._coords_to_widget(event)
        view_w = self.scene_widget.frame.width
        view_h = self.scene_widget.frame.height

        cam = self.scene_widget.scene.camera
        near = cam.unproject(mx_scene, my_scene, 0.0, view_w, view_h)
        far  = cam.unproject(mx_scene, my_scene, 0.99, view_w, view_h)
        origin = np.array(near, dtype=float)
        dirv = np.array(far, dtype=float) - origin
        nrm = np.linalg.norm(dirv)
        if nrm < 1e-8:
            return origin, None
        return origin, dirv / nrm
    
    def _camera_under_ray(self, origin, direction, max_vertex_dist=0.5):
        """
        Return cam_idx under ray or None.
        - origin, direction must be numpy arrays; direction should be normalized.
        - We first do a bounding sphere reject, then a vertex distance check.
        """
        best_idx = None
        best_dist = float("inf")

        for cam_idx, d in self.camera_frustums.items():
            center = d["center"]
            radius = d["radius"]

            # project center onto ray (scalar)
            to_center = center - origin
            proj = np.dot(to_center, direction)

            # quick reject: if projected closest approach is far from center by more than radius + margin
            closest_point = origin + proj * direction
            if np.linalg.norm(closest_point - center) > (radius + max_vertex_dist):
                continue

            # fine check: distance to frustum vertices (distance to infinite line)
            verts = d["verts"]
            # direction should be normalized -> no division needed
            dists = np.linalg.norm(np.cross(verts - origin, direction), axis=1)
            min_dist = dists.min()
            if min_dist < best_dist:
                best_dist = min_dist
                best_idx = cam_idx

        if best_idx is not None and best_dist <= max_vertex_dist:
            return best_idx
        return None

    def _on_camera_selected(self, cam_idx):
        # Load original image
        img_path = self.imgs_paths[cam_idx]
        img_cv2 = None
        if img_path is not None:
            #img = o3d.io.read_image(img_path)
            img_cv2 = cv2.imread(img_path)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img = o3d.geometry.Image(img_cv2)
            self.cam_original_view.update_image(img)
    
    def _coords_to_widget(self, event):
        """Convert absolute event coords to widget-local coords (pixels from top-left of widget)."""
        mx = event.x - self.scene_widget.frame.x
        my = event.y - self.scene_widget.frame.y
        return mx, my


    def on_close(self):
        return True

    def run(self):
        self.app.run()