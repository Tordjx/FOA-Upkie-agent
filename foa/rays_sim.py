import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
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
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
import gymnasium as gym
class RaysSimWrapper(gym.Wrapper) : 
    def __init__(self, env,image_every = 10):
        super(RaysSimWrapper, self).__init__(env)
        self.rays_sim = RaysSim()
        self.image_every = image_every
        self.obstacle_points = None
        self.image_count = 0
    def reset(self, **kwargs):
        s,i = self.env.reset(**kwargs)
        position = i['spine_observation']['sim']['base']['position']
        w,x,y,z = i['spine_observation']['sim']['base']['orientation'] #wxyz
        quaternion = np.array([x,y,z,w]) #xyzw
        i['obstacle_points'] = self.rays_sim.get_ray_points_local_frame( position, quaternion)
        self.image_count = 0
        return s,i
    def step(self, action):
        s,r,d,t,i = self.env.step(action)
        if self.image_count % self.image_every == 0:
            position = i['spine_observation']['sim']['base']['position']
            w,x,y,z = i['spine_observation']['sim']['base']['orientation'] #wxyz
            quaternion = np.array([x,y,z,w]) #xyzw
            i['obstacle_points']= self.rays_sim.get_ray_points_local_frame( position, quaternion)
            self.obstacle_points = i['obstacle_points']
        else:
            i['obstacle_points'] = self.obstacle_points
        self.image_count += 1
        return s,r,d,t,i
class RaysSim:
    def __init__(self):
        # Load Gaussian splatting model and point cloud
        self.gaussians = GaussianModel(3)
        model_path = "data"
        self.gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))

        # Camera parameters
        self.fovx = 69 * (np.pi / 180)
        self.fovy = 54 * (np.pi / 180)
        self.znear = 0.02
        self.zfar = 100
        self.width = 128
        self.height = 128

        # Ray shooting params
        self.N = 20  # Number of rays horizontally
        self.radius = 2  # Patch half-size for median filtering

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render_depth(self, position, quaternion):
        """
        Render depth image from given position and quaternion.

        position: array-like, shape (3,) - x, y, z (height)
        quaternion: array-like, shape (4,) - [x, y, z, w] quaternion (xyzw convention)
        """
        posx, posy, posz = position
        x, y, z, w = quaternion

        C = [posx, posy, posz]
        R = Rotation.from_quat([x, y, z, w]).as_matrix()

        # Adjust camera orientation to match rendering conventions
        adjust_rot = Rotation.from_euler("xyz", [np.pi / 2, np.pi, np.pi / 2]).as_matrix()
        R = R @ adjust_rot

        T = -R.T @ C

        view = Camera(
            colmap_id=0,
            R=R,
            T=T,
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
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            rendering = render(view, self.gaussians, pipeline, background)
            depth = rendering["surf_depth"].cpu().numpy()

        return depth.squeeze(0)

    def get_rays(self, position, quaternion):
        """
        Given a 3D position and orientation quaternion, render depth and
        return median depth values on a horizontal slice (row) of the depth image.
        """
        depth = self.render_depth(position, quaternion)

        frame_height, frame_width = depth.shape

        # Focal length in pixels (approximate)
        f_y = frame_height / (2 * np.tan(self.fovy / 2))
        c_y = frame_height / 2

        # Compute pixel row corresponding to camera optical axis elevation:
        # We find vertical angle of the camera direction vector to image plane center
        # The elevation angle = arcsin of the z-axis of rotation matrix
        R = Rotation.from_quat(quaternion).as_matrix()
        camera_z_axis = R[:, 2]  # camera's forward vector
        elevation = np.arcsin(camera_z_axis[2])  # vertical angle of optical axis

        pixel_row = int(np.clip(f_y * np.tan(-elevation) + c_y, 0, frame_height - 1))

        # Precompute circular mask for median filtering
        Y, X = np.ogrid[-self.radius : self.radius + 1, -self.radius : self.radius + 1]
        circular_mask = X**2 + Y**2 <= self.radius**2

        cols = np.linspace(0, frame_width - 1, self.N, dtype=int)

        median_depths = []
        for col in cols:
            # Define patch boundaries
            row_start = max(pixel_row - self.radius, 0)
            row_end = min(pixel_row + self.radius + 1, frame_height)
            col_start = max(col - self.radius, 0)
            col_end = min(col + self.radius + 1, frame_width)

            patch = depth[row_start:row_end, col_start:col_end]

            # Adjust mask if patch is smaller at edges
            mask_r_start = self.radius - (pixel_row - row_start)
            mask_r_end = mask_r_start + patch.shape[0]
            mask_c_start = self.radius - (col - col_start)
            mask_c_end = mask_c_start + patch.shape[1]

            patch_mask = circular_mask[mask_r_start:mask_r_end, mask_c_start:mask_c_end]

            # Apply mask and compute median of valid pixels inside circle
            median_val = np.median(patch[patch_mask])
            median_depths.append(float(median_val))

        return median_depths, pixel_row, cols
    def get_ray_points_local_frame(self, position, quaternion):
        """
        Return 3D points (in meters) in the robot's local frame (x forward, y left, z up)
        corresponding to rays across the rendered depth image.
        """
        
        median_depths, pixel_row, cols = self.get_rays(position, quaternion)

        # Intrinsics
        fx = self.width / (2 * np.tan(self.fovx / 2))
        fy = self.height / (2 * np.tan(self.fovy / 2))
        cx = self.width / 2
        cy = self.height / 2

        points = []
        for u, depth_val in zip(cols, median_depths):
            v = pixel_row

            # From image coordinates (u, v) + depth to camera coordinates (OpenCV: z forward)
            z_cam = depth_val
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy

            # Convert to robot frame: x forward, y left, z up
            x_robot = z_cam
            y_robot = -x_cam
            z_robot = -y_cam

            points.append(np.array([x_robot, y_robot])) #2d points

        return np.array(points)
