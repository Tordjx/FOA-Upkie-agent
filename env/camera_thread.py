import threading

import os
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings
from scipy.spatial.transform import Rotation as R
config = EnvSettings()
zmin = -0.2 * 1000   # meters
zmax = 1 * 1000 
xmax = 1   * 1000   # forward limit

Rmat = np.array([
    [ 0,  0,  1],   # Z -> Y
    [-1,  0,  0],   # -X -> Z
    [ 0, 1,  0],   # -Y -> X
])
def set_self_affinity(core):
    tid = threading.get_native_id()  # Linux TID
    os.sched_setaffinity(tid, {core})
#@profile
def get_points(points, pitch):
    # Precompute sin/cos
    c = np.cos(-pitch)
    s = np.sin(-pitch)

    # Pitch undo matrix
    Rp = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)
    Rtotal = Rp @ Rmat
    points = points@Rtotal.T
    x = points[:, 0]
    z = points[:, 2]

    mask = (x > 0.0) & (x < xmax) & (z > zmin) & (z < zmax)
    points = points[mask]

    n = min(30, len(points))
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)

    idx = np.random.choice(len(points), size=n, replace=False)
    
    return points[idx, :2]


class CameraThread:
    def __init__(self, intrinsics, pipeline, pcl_queue, imuQueue,device, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.device = device
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
    #@profile
    def run(self):
        set_self_affinity(1)
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
            points = inPcl.getPoints()
            xy = get_points(points, pitch) / 1000.0
            with self.lock:
                self.latest_points = xy

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_points
