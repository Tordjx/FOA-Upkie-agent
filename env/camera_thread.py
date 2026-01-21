import threading
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings
from scipy.spatial.transform import Rotation as R
config = EnvSettings()

def process_depth_for_obstacles(depth_m, quat, fx, fy, cx, cy,
                                z_min=0.0, z_max=3.0,
                                grid_shape=(15,2), percentile=10, max_depth = 3):
    """
    Process a depth image for obstacle avoidance.
    
    Parameters:
        depth_m : H x W np.ndarray
            Depth image in meters.
        quat : [w, x, y, z]
            Camera orientation quaternion (DepthAI order).
        fx, fy, cx, cy : float
            Camera intrinsics.
        z_min, z_max : float
            Min/max world Z to keep in depth.
        grid_shape : (rows, cols)
            Grid to split depth image for obstacle points.
        percentile : int
            Percentile in each grid cell to represent obstacle.

    Returns:
        depth_filtered : H x W np.ndarray
            Depth image filtered along world Z.
        obs_points : N x 2 np.ndarray
            Representative obstacle points in world XY (top-down).
    """
    h, w = depth_m.shape
    r_rows, r_cols = grid_shape
    row_splits = np.linspace(0, h, r_rows+1, dtype=int)
    col_splits = np.linspace(0, w, r_cols+1, dtype=int)

    # --- Convert quaternion to rotation matrix ---
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # x,y,z,w
    R_cam2world = r.as_matrix()

    # --- Back-project all pixels to camera coordinates ---
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    Xc = (uu - cx) * depth_m / fx
    Yc = (vv - cy) * depth_m / fy
    Zc = depth_m
    points_cam = np.stack([Xc, Yc, Zc], axis=-1)  # H x W x 3

    # --- Rotate to world frame ---
    points_world = points_cam @ R_cam2world.T  # H x W x 3

    # --- Filter depth by world Z ---
    mask = (points_world[:,:,2] < z_min) | (points_world[:,:,2] > z_max)
    depth_filtered = depth_m.copy()
    depth_filtered[mask] = max_depth 

    # --- Compute obstacle points in XY ---
    obs_points = []
    for i in range(r_rows):
        for j in range(r_cols):
            patch = depth_filtered[row_splits[i]:row_splits[i+1],
                                   col_splits[j]:col_splits[j+1]]
            if np.count_nonzero(patch) == 0:
                continue
            # Low-percentile depth in this patch
            z_cam = np.percentile(patch[patch>0], percentile)

            # Patch center pixel
            u_center = (col_splits[j] + col_splits[j+1]) / 2
            v_center = (row_splits[i] + row_splits[i+1]) / 2

            # Back-project to camera coordinates
            Xc = (u_center - cx) * z_cam / fx
            Yc = (v_center - cy) * z_cam / fy
            Zc = z_cam
            pt_cam = np.array([Xc, Yc, Zc])

            # Rotate to world
            pt_world = pt_cam
            if pt_world[-1] <max_depth : 
                obs_points.append(pt_world[1:3])

    obs_points = np.array(obs_points)
    return depth_filtered, obs_points


class CameraThread:
    def __init__(self, intrinsics, pipeline, stereoOut, imuQueue,device, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.device = device
        self.intrinsics = intrinsics 
        self.pipeline = pipeline 
        self.stereoOut = stereoOut
        self.imuQueue = imuQueue
        self.fps = fps
        self.rate_limiter = RateLimiter(frequency = fps)
        self.latest_points = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self.run)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        self.pipeline.start()

        # Enable IR dots & flood
        self.device.setIrLaserDotProjectorIntensity(1.0)
        self.device.setIrFloodLightIntensity(0.0)

        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]  # fx, fy
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]  # cx, cy
        while self.running:
            depthIn = self.stereoOut.get()
        
            q = self.imuQueue.get().packets[0].rotationVector
            w, x, y, z = q.real, q.i, q.j, q.k
            quat = [w,x,y,z]
            depthFrame = depthIn.getFrame()                  # uint16 mm
            depthMeters = depthFrame.astype(np.float32) / 1000.0
            # Fill missing values with max depth
            depthFilled = depthMeters.copy()

            max_depth =2
            depthFilled[depthFilled == 0] = max_depth


            # Normalize for visualization
            dispVis = np.clip(depthFilled, 0, max_depth)
            dispVis, obs_points = process_depth_for_obstacles(dispVis, quat, z_min=-0.2, z_max=1.5, fx=fx, fy=fy, cx=cx, cy=cy,max_depth=max_depth)
            # Store safely
            with self.lock:
                self.latest_points = obs_points

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_points
