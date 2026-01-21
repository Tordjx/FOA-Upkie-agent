import depthai as dai
import gin
from gymnasium import Wrapper
import torch
from autoencoder import AutoEncoder
import numpy as np
gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings

config = EnvSettings()
import gymnasium as gym
from foa.monocular_thread import MonocularThread


def create_pipeline(blob_path = "autoencoder.blob"):
    # Create the pipeline
    pipeline = dai.Pipeline()

    # Define a color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)  # Original resolution
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    nn = pipeline.create(dai.node.NeuralNetwork)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    # Define Image Manipulation for cropping and grayscale conversion
    manip = pipeline.createImageManip()

    # Configure resizing settings
    manip.initialConfig.setResize(config.height, config.width)
    manip.setKeepAspectRatio(True)

    # Link camera preview output to ImageManip input
    cam_rgb.preview.link(manip.inputImage)

    # Optionally, create an XLinkOut to stream the grayscale image to the host
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    xout_nn.setStreamName("nn")
    manip.out.link(xout.input)

    nn.setBlobPath(blob_path)
    manip.out.link(nn.input)

    nn.out.link(xout_nn.input)

    return dai.Device(pipeline)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonocularWrapper(Wrapper):
    def __init__(self, env, image_every=10):
        super().__init__(env=env)
        self.device = create_pipeline()
        
        self.camera_thread = MonocularThread(camera = self.device, fps=10 )
        self.camera_thread.start()

    def step(self, action):
        s, r, d, t, i = self.env.step(action)
        pitch = i["spine_observation"]["base_orientation"]["pitch"]
        self.camera_thread.set_pitch(pitch)
        self.obstacle_points = self.camera_thread.get_latest()
        print(self.obstacle_points)
        i["obstacle_points"] = self.obstacle_points
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        pitch = i["spine_observation"]["base_orientation"]["pitch"]
        self.camera_thread.set_pitch(pitch)
        self.obstacle_points = self.camera_thread.get_latest()
        i["obstacle_points"] = self.obstacle_points
        return s, i
