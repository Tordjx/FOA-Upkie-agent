import threading
import gin
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings

config = EnvSettings()

from autoencoder import AutoEncoder
model =  AutoEncoder(input_shape=(3, 128,128), z_size=32)
model.load_state_dict(torch.load('autoencoder.pth', map_location = torch.device('cpu')))
def get_features(device):
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)
    q_img = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    in_rgb = None 
    while in_rgb is None : 
        in_rgb = q_img.tryGet()
    in_nn = None
    while in_nn is None:
        in_nn = q_nn.tryGet()
    with torch.no_grad():
        return model.encode(torch.from_numpy(in_rgb.getCvFrame()[..., ::-1].copy()).moveaxis(-1,0).unsqueeze(0).float()/(255)).squeeze(0)
    """
    print(model.encode(torch.from_numpy(in_rgb.getCvFrame()[..., ::-1].copy()).moveaxis(-1,0).unsqueeze(0).float()/(255)))
    features = np.array(in_nn.getFirstLayerFp16())  # or getLayerFp16("layer_name")
    print(features)
    return torch.from_numpy(features.astype(np.float32))"""



class CameraThread:
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
        self.latest_encoded = None
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
        while self.running:
            # Get image from camera
            encoded = get_features(self.camera)

            # Store safely
            with self.lock:
                self.latest_encoded = encoded

            self.rate_limiter.sleep()

    def get_latest(self):
        with self.lock:
            return self.latest_encoded
