#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation as R
# IR settings
dot_intensity = 1.0
flood_intensity = 0.0

# Create device

import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R

def process_depth_for_obstacles(depth_m, quat, fx, fy, cx, cy,
                                z_min=0.0, z_max=3.0,
                                grid_shape=(15,2), percentile=10, max_depth = 3):
    """
    Process a depth image for obstacle avoidance.
    
    Parameters:
        depth_m : H x W np.ndarray
            Depth image in meters.
        quat : [w, x, y, z]
            Camera orientation quaternion (DepthAI order).
        fx, fy, cx, cy : float
            Camera intrinsics.
        z_min, z_max : float
            Min/max world Z to keep in depth.
        grid_shape : (rows, cols)
            Grid to split depth image for obstacle points.
        percentile : int
            Percentile in each grid cell to represent obstacle.

    Returns:
        depth_filtered : H x W np.ndarray
            Depth image filtered along world Z.
        obs_points : N x 2 np.ndarray
            Representative obstacle points in world XY (top-down).
    """
    h, w = depth_m.shape
    r_rows, r_cols = grid_shape
    row_splits = np.linspace(0, h, r_rows+1, dtype=int)
    col_splits = np.linspace(0, w, r_cols+1, dtype=int)

    # --- Convert quaternion to rotation matrix ---
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # x,y,z,w
    R_cam2world = r.as_matrix()

    # --- Back-project all pixels to camera coordinates ---
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    Xc = (uu - cx) * depth_m / fx
    Yc = (vv - cy) * depth_m / fy
    Zc = depth_m
    points_cam = np.stack([Xc, Yc, Zc], axis=-1)  # H x W x 3

    # --- Rotate to world frame ---
    points_world = points_cam @ R_cam2world.T  # H x W x 3

    # --- Filter depth by world Z ---
    mask = (points_world[:,:,2] < z_min) | (points_world[:,:,2] > z_max)
    depth_filtered = depth_m.copy()
    depth_filtered[mask] = max_depth 

    # --- Compute obstacle points in XY ---
    obs_points = []
    for i in range(r_rows):
        for j in range(r_cols):
            patch = depth_filtered[row_splits[i]:row_splits[i+1],
                                   col_splits[j]:col_splits[j+1]]
            if np.count_nonzero(patch) == 0:
                continue
            # Low-percentile depth in this patch
            z_cam = np.percentile(patch[patch>0], percentile)

            # Patch center pixel
            u_center = (col_splits[j] + col_splits[j+1]) / 2
            v_center = (row_splits[i] + row_splits[i+1]) / 2

            # Back-project to camera coordinates
            Xc = (u_center - cx) * z_cam / fx
            Yc = (v_center - cy) * z_cam / fy
            Zc = z_cam
            pt_cam = np.array([Xc, Yc, Zc])

            # Rotate to world
            pt_world = pt_cam
            obs_points.append(pt_world[1:3])

    obs_points = np.array(obs_points)
    return depth_filtered, obs_points


device = dai.Device()
intrinsics = device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
intrinsics  = np.array(intrinsics)

def plot_obstacles(obs_points, window_size=(500,500), scale=100):
    """
    Plot 2D obstacle points in top-down view using OpenCV.
    Centers the points in the window and ensures visibility.
    """
    h, w = window_size
    topdown = np.zeros((h,w,3), dtype=np.uint8)

    # Compute window center
    center_x = w // 2
    center_y = h // 2

    # Draw each point
    for p in obs_points:
        # p = [X, Y] in meters
        x_disp = int(center_x + p[0]*scale)
        y_disp = int(center_y - p[1]*scale)  # Y inverted: forward is up

        # Only draw points inside the window
        if 0 <= x_disp < w and 0 <= y_disp < h:
            cv2.circle(topdown, (x_disp, y_disp), 5, (0,0,255), -1)

    cv2.imshow("Top-Down Obstacles", topdown)
    cv2.waitKey(1)

with dai.Pipeline(device) as pipeline:
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
    # Start pipeline
    pipeline.start()

    # Enable IR dots & flood
    device.setIrLaserDotProjectorIntensity(dot_intensity)
    device.setIrFloodLightIntensity(flood_intensity)

    # ---------- Visualization ----------
    colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colorMap[0] = [0, 0, 0]
    # After pipeline.start()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # fx, fy
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # cx, cy
    while pipeline.isRunning():
        depthIn = stereoOut.tryGet()
        

        if depthIn:
            q = imuQueue.get().packets[0].rotationVector
            w, x, y, z = q.real, q.i, q.j, q.k
            quat = [w,x,y,z]
            r = R.from_quat([x,y,z,w])  # scipy expects x,y,z,w

            # Get Euler angles in degrees, using standard aerospace convention
            roll, pitch, yaw = r.as_euler('xyz', degrees=True)
            depthFrame = depthIn.getFrame()                  # uint16 mm
            depthMeters = depthFrame.astype(np.float32) / 1000.0

            # Fill missing values with max depth
            depthFilled = depthMeters.copy()

            max_depth =2
            depthFilled[depthFilled == 0] = max_depth


            # Normalize for visualization
            dispVis = np.clip(depthFilled, 0, max_depth)
            dispVis, obs_points = process_depth_for_obstacles(dispVis, quat, z_min=-0.2, z_max=1.5, fx=fx, fy=fy, cx=cx, cy=cy,max_depth=max_depth)
            dispVis = (dispVis / max_depth * 255).astype(np.uint8)
            dispColor = cv2.applyColorMap(dispVis, cv2.COLORMAP_JET)
            # Optional: numeric depth at center
            h, w = depthFilled.shape
            centerDepth = depthFilled[h//2, w//2]
            cv2.putText(dispColor, f"{centerDepth:.2f} m",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)


            cv2.putText(dispColor, f"Roll: {roll:.1f} deg",  (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(dispColor, f"Pitch: {pitch:.1f} deg", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(dispColor, f"Yaw: {yaw:.1f} deg",     (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Depth (smooth, filled, IR ON)", dispColor)
            plot_obstacles(obs_points)
        if cv2.waitKey(1) == ord('q'):
            pipeline.stop()
            break


cv2.destroyAllWindows()
