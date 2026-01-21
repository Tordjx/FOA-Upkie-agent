import importlib
import os
import sys

def import_all_modules_from_package(package_name):
    splatting_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../third_party/2d-gaussian-splatting",
        )
    )
    if splatting_dir not in sys.path:
        sys.path.append(splatting_dir)
    package_dir = os.path.join(splatting_dir, package_name)
    for module_name in os.listdir(package_dir):
        if module_name.endswith(".py") and module_name != "__init__.py":
            module_name = module_name[:-3]  # Remove the .py extension
            importlib.import_module(f"{package_name}.{module_name}")


# Replace 'your_package_name' with the actual name of the package
import_all_modules_from_package("gaussian_renderer")
import_all_modules_from_package("scene")
import_all_modules_from_package("utils")
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import torch
import torchvision
from gaussian_renderer import GaussianModel, render
from gymnasium import Wrapper
from scene.cameras import Camera
from scipy.spatial.transform import Rotation as rot
from upkie_description import load_in_pinocchio




def euler_to_rotation_matrix(euler_angles):
    return rot.from_euler("xyz", euler_angles).as_matrix()

def quaternion_to_rotation_matrix(quaternion):
    return rot.from_quat(quaternion).as_matrix()


class UpkieVisionWrapper(Wrapper):
    def __init__(
        self,
        env,
        model_path,
        fovx,
        fovy,
        width=256,
        height=256,
        image_every=1,
        window=False,
    ):
        super().__init__(env=env)
        self.height = height
        self.width = width
        self.gaussians = GaussianModel(3)
        self.gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))
        self.fovx, self.fovy = fovx, fovy
        self.znear, self.zfar = 0.02, 100
        self.window = window
        if window:
            self.window_name = str(np.random.random())
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.robot = load_in_pinocchio(
            root_joint=pin.JointModelFreeFlyer(), variant="camera"
        )
        pin.framesForwardKinematics(self.robot.model, self.robot.data, self.robot.q0)
        self.camera_translation = (
            self.robot.data.oMf[self.robot.model.getFrameId("camera_eye")].translation
        ) @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.image_every = image_every
        self.image_count = 0

    def step(self, action):
        s, r, d, t, i = self.env.step(action)
        if self.image_count % self.image_every == 0:
            image, depth = self.display_image(i["spine_observation"])
            self.image = image
            self.depth = depth
        else:
            image = self.image
            depth = self.depth
        i["image"] = image
        i["depth"] = depth
        h, w = image.shape[:2]
        self.image_count += 1
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset()
        image, depth = self.display_image(i["spine_observation"])
        self.image = image
        self.depth = depth
        i["image"] = image
        i["depth"] = depth
        self.image_count = 0
        h, w = image.shape[:2]
        return s, i

    def display_image(self, info, C=None):

        self.R, self.upkie_pos = (
            info["sim"]["base"]["orientation"],
            info["sim"]["base"]["position"],
        )
        w, x, y, z = self.R
        self.R = x, y, z, w
        self.upkie_rot, self.upkie_pos = np.array(self.R), np.array(self.upkie_pos)
        self.R = self.upkie_rot
        self.upkie_rot = quaternion_to_rotation_matrix(
            self.R
        ) @ euler_to_rotation_matrix((-np.pi / 2, 0, np.pi / 2))
        self.upkie_pos += (
            quaternion_to_rotation_matrix(self.R)
            @ self.camera_translation
            @ euler_to_rotation_matrix((0, 0, np.pi / 2))
        )
        self.R = quaternion_to_rotation_matrix(self.R) @ euler_to_rotation_matrix(
            (-np.pi / 2, 0, -np.pi / 2)
        )
        self.C = self.upkie_pos
        self.T = -self.R.T @ self.C
        return self.give_image(info)

    def give_upkie_image(self, info):
        joints = [
            "left_hip",
            "left_knee",
            "left_wheel",
            "right_hip",
            "right_knee",
            "right_wheel",
        ]
        orientation, position = (
            info["sim"]["base"]["orientation"],
            info["sim"]["base"]["position"],
        )
        w, x, y, z = orientation
        posx, poy, poz = position
        q = np.array(
            [
                posx,
                poy,
                poz,
                x,
                y,
                z,
                w,
            ]
            + [info["servo"][joint]["position"] for joint in joints]
        )
        rgb_image = self.viewer.get_screenshot(requested_format="RGB") / 255
        depth = self.viewer.get_depth_screenshot()
        return rgb_image, depth

    def give_image(self, info):
        view = Camera(
            colmap_id=0,
            R=self.R,
            T=self.T,
            FoVx=self.fovx,
            FoVy=self.fovy,
            image=torch.zeros((3, self.width, self.height)),
            gt_alpha_mask=None,
            image_name="",
            uid=0,
        )

        class PipelineParams:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False
                self.depth_ratio = 0.0

        pipeline = PipelineParams()
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            rendering = render(view, self.gaussians, pipeline, background)
            image = rendering["render"].permute(1, 2, 0)
            depth = (
                rendering["surf_depth"]
                .permute(1, 2, 0)
                .reshape(self.height, self.width, 1)
            )
        image = image.moveaxis(-1, 0).cpu().numpy()
        if self.window:

            window_image = (255 * np.moveaxis(image, 0, -1)).astype(np.uint8)

            upscaled_image = cv2.resize(
                window_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR
            )
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, upscaled_image)
            cv2.waitKey(1)
        return image, depth.cpu().numpy().reshape((self.height, self.width))
