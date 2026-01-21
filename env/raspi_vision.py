import depthai as dai
import gin
from gymnasium import Wrapper
import torch
import numpy as np
gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings

config = EnvSettings()
from env.camera_thread import CameraThread


def create_pipeline():
    device = dai.Device()
    intrinsics = device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
    intrinsics  = np.array(intrinsics)

    pipeline = dai.Pipeline(device) 
    # ---------- Cameras ----------
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    monoLeftOut  = monoLeft.requestFullResolutionOutput(fps=15, type=dai.ImgFrame.Type.GRAY8)
    monoRightOut = monoRight.requestFullResolutionOutput(fps=15, type=dai.ImgFrame.Type.GRAY8)

    # ---------- Stereo ----------
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setDisparityShift(True)
    stereo.initialConfig.costMatching.enableCompanding = True
    
    stereo.initialConfig.setConfidenceThreshold(0)
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)
    
    stereo.setRectification(True)
    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # ---------- Depth output ----------
    stereoOut = stereo.depth.createOutputQueue(maxSize=1, blocking=False)  # depth in mm

    imu = pipeline.create(dai.node.IMU)

    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 480)

    imuQueue = imu.out.createOutputQueue(maxSize=1, blocking=False)
    return intrinsics, pipeline, stereoOut, imuQueue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RaspiImageWrapper(Wrapper):
    def __init__(self, env, image_every=10):
        super().__init__(env=env)
        intrinsics, pipeline, stereoOut, imuQueue= create_pipeline()
        self.camera_thread = CameraThread(
            intrinsics = intrinsics, pipeline = pipeline, stereoOut = stereoOut, imuQueue = imuQueue, fps=10
        )
        self.camera_thread.start()

    def step(self, action):
        s, r, d, t, i = self.env.step(action)
        i['obstacle_points'] = self.camera_thread.get_latest()
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        i['obstacle_points'] = self.camera_thread.get_latest()
        return s, i
