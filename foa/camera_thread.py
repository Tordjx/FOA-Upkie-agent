import threading

import os
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter
from foa.rays_processor import RaysProcessor

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings
from scipy.spatial.transform import Rotation as R
config = EnvSettings()
def set_self_affinity(core):
    tid = threading.get_native_id()  # Linux TID
    os.sched_setaffinity(tid, {core})


class CameraThread:
    def __init__(self, intrinsics, pipeline, stereo_queue, imuQueue,device, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.device = device
        self.intrinsics = intrinsics 
        self.pipeline = pipeline 
        self.stereo_queue = stereo_queue
        self.imuQueue = imuQueue
        self.fps = fps
        self.rate_limiter = RateLimiter(frequency = fps)
        self.latest_points = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self.run)
        self.rays_processor = None

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
            inDepth = self.stereo_queue.get()
            imuMsg = self.imuQueue.get()
            if inDepth is None or imuMsg is None:
                continue

            q = imuMsg.packets[0].rotationVector
            quat = [q.i, q.j, q.k, q.real]
            depth = inDepth.getFrame()
            if self.rays_processor is None :
                frame_height, frame_width = depth.shape
                fx = self.intrinsics[0, 0]
                fy = self.intrinsics[1, 1]

                fovx = 2 * np.arctan(frame_width / (2 * fx))
                fovy = 2 * np.arctan(frame_height / (2 * fy))
                self.rays_processor  = RaysProcessor(fovx, fovy , 1)
            xy = self.rays_processor.get_ray_points_local_frame(depth, quat)
            with self.lock:
                self.latest_points = xy

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_points
