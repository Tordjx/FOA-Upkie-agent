import threading
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings
from scipy.spatial.transform import Rotation as R
config = EnvSettings()
zmin = -0.2    # meters
zmax = 1
xmax = 1    # forward limit

Rmat = np.array([
    [ 0,  0,  1],   # Z -> Y
    [-1,  0,  0],   # -X -> Z
    [ 0, 1,  0],   # -Y -> X
])
def get_points(points, pitch) : 

    # DepthAI → robot frame
    points = (Rmat @ points.T).T

    # ---------- REMOVE PITCH ----------
    points = (Rp := np.array([
        [ np.cos(-pitch), 0, np.sin(-pitch)],
        [ 0,              1, 0             ],
        [-np.sin(-pitch), 0, np.cos(-pitch)]
    ]) @ points.T).T

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x > 0.0) &
        (x < xmax) &
        (z > zmin) &
        (z < zmax)
    )

    points = points[mask]
    # 2D robot plane
    n = min(30, len(points))   # safety if fewer than 30
    idx = np.random.choice(len(points), size=n, replace=False)
    xy = points[idx, :2]
    return xy

class CameraThread:
    def __init__(self, intrinsics, pipeline, pcl_queue, imuQueue, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.intrinsics = intrinsics 
        self.pipeline = pipeline 
        self.pcl_queue = pcl_queue
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
        while self.running:
            inPcl = self.pcl_queue.get()
            imuMsg = self.imuQueue.get()
            if inPcl is None or imuMsg is None:
                continue

            q = imuMsg.packets[0].rotationVector
            rot = R.from_quat([q.i, q.j, q.k, q.real])
            pitch, _ ,_ = rot.as_euler("xyz", degrees=False)
            pitch = pitch + np.pi/2
            # Nx3 float32 (meters)
            points = inPcl.getPoints() / 1000.0
            xy = get_points(points, pitch)
            with self.lock:
                self.latest_points = xy

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_points
