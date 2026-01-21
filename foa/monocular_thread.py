import threading
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings

config = EnvSettings()

import zmq

    

def get_depth(device):
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)
    q_img = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    in_rgb = None 
    while in_rgb is None : 
        in_rgb = q_img.tryGet()
    in_nn = None
    while in_nn is None:
        in_nn = q_nn.tryGet()
    depth = np.array(in_nn.getFirstLayerFp16())  # or getLayerFp16("layer_name")
    return torch.from_numpy(depth.astype(np.float32)),in_rgb.getCvFrame()



class MonocularThread:
    def __init__(self, camera, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.camera = camera
        self.fps = fps
        self.dt = 1.0 / fps
        self.rate_limiter = RateLimiter(frequency = fps)
        self.latest_points = np.zeros((20,2))
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self.run)
        self.fovx = 69 * (np.pi / 180) 
        self.fovy = 54 * (np.pi / 180) 
        self.cx = 128 // 2 
        self.cy = 128 // 2
        self.n_points = 20
        self.pitch = 0

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:8081")  # Change port if needed

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            # Get image from camera
            depth , image= get_depth(self.camera)
            depth = np.exp(depth.view(128,128))
            self.socket.send_pyobj(depth)  # Publish numpy array (pickled)
            # Store safely
            with self.lock:
                latest_points = self.depth_to_points(depth)
                latest_points[:,1] *= -1
                self.latest_points = latest_points

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_points
    def set_pitch(self,pitch):
        self.pitch = pitch
    def depth_to_points(self, depth, window_height=5, window_width=5):
        """
        depth: HxW torch tensor depth map
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu()  # ensure on CPU
        h, w = depth.shape

        # Compute center row corresponding to camera forward
        center_row = int(self.cy + (self.pitch / (self.fovy / 2)) * self.cy)
        center_row = np.clip(center_row, 0, h - 1)

        # Sample points along width evenly
        center_cols = np.linspace(0, w - 1, self.n_points).astype(int)

        Xc_list, Yc_list, Zc_list = [], [], []

        for col in center_cols:
            # Define vertical + horizontal window
            row_start = max(center_row - window_height, 0)
            row_end   = min(center_row + window_height + 1, h)
            rows = torch.arange(row_start, row_end)

            col_start = max(col - window_width, 0)
            col_end   = min(col + window_width + 1, w)
            cols = torch.arange(col_start, col_end)

            # Extract sub-window
            z_box = depth[rows[:, None], cols[None, :]]  # torch tensor

            # Closest nonzero depth
            valid_depths = z_box[z_box > 0]
            if valid_depths.numel() > 0:
                z_sel = valid_depths.min().item()
            else:
                z_sel = float("nan")

            # Project pixel → normalized camera coords
            x_norm = (col - self.cx) / self.cx * np.tan(self.fovx / 2)
            y_norm = (center_row - self.cy) / self.cy * np.tan(self.fovy / 2)

            Xc_list.append(z_sel * x_norm)
            Yc_list.append(z_sel * y_norm)
            Zc_list.append(z_sel)

        # Stack points in camera frame
        pts_cam = np.vstack((Xc_list, Yc_list, Zc_list))

        # Rotate for pitch
        c, s = np.cos(self.pitch), np.sin(self.pitch)
        R_pitch = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        pts_cam = R_pitch @ pts_cam

        # Camera → robot frame
        R_cam_to_robot = np.array([[0, 0, 1],
                                [-1, 0, 0],
                                [0, -1, 0]])
        
        pts_robot = R_cam_to_robot @ pts_cam
        pts_robot = (pts_robot.T)[:,:2]
        print(pts_robot.shape)
        pts_robot = pts_robot[(pts_robot**2).sum(1)<1.5**2]
        return pts_robot