import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class RaysProcessor:
    def __init__(self,fovx, fovy, max_depth, N=30, radius=10):
        self.fovx = 100*np.pi/180#fovx
        self.fovy = fovy
        self.N = N
        self.radius = radius
        self.max_depth = max_depth


    def get_rays(self,depth , quaternion):
        frame_height, frame_width = depth.shape

        # Focal length in pixels (approximate)
        f_y = frame_height / (2 * np.tan(self.fovy / 2))
        c_y = frame_height / 2

        R = Rotation.from_quat(quaternion).as_matrix()
        camera_z_axis = R[:, 2]  # camera's forward vector
        elevation = -np.arcsin(camera_z_axis[2])  # vertical angle of optical axis 
        print('ELEVATION :',elevation )

        pixel_row = int(np.clip(f_y * np.tan(-elevation) + c_y, 0, frame_height - 1))
        print(pixel_row,depth.shape)
        cols = np.linspace(0, frame_width - 1, self.N, dtype=int)

        median_depths = []
        for col in cols:
            row_start = max(pixel_row - self.radius, 0)
            row_end = min(pixel_row + self.radius + 1, frame_height)
            col_start = max(col - self.radius, 0)
            col_end = min(col + self.radius + 1, frame_width)

            patch = depth[row_start:row_end, col_start:col_end]
            median_val = np.quantile(patch, 0.1)
            median_depths.append(float(median_val))

        return median_depths, pixel_row, cols

    def get_ray_points_local_frame(self, depth , quaternion):
        frame_height, frame_width = depth.shape
        depth = np.where(depth==0 , self.max_depth, depth/1000)
        median_depths, pixel_row, cols = self.get_rays(depth,quaternion)
        
        fx = frame_width / (2 * np.tan(self.fovx / 2))
        fy = frame_height / (2 * np.tan(self.fovy / 2))
        cx = frame_width / 2
        cy = frame_height / 2

        points = []
        for u, depth_val in zip(cols, median_depths):
            if depth_val < self.max_depth: 
                v = pixel_row

                z_cam = depth_val
                x_cam = (u - cx) * z_cam / fx
                y_cam = (v - cy) * z_cam / fy

                # Robot frame: x forward, y left
                x_robot = z_cam
                y_robot = -x_cam

                points.append([x_robot, y_robot])

        return np.array(points)
