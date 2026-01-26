import cv2
import numpy as np
import matplotlib.pyplot as plt
from foa.camera_thread import CameraThread
from env.raspi_vision import create_pipeline
from foa.rays_processor import RaysProcessor

intrinsics, pipeline, stereo_queue, imuQueue, device = create_pipeline()
device.setIrLaserDotProjectorIntensity(1.0)
device.setIrFloodLightIntensity(0.0)

rays_processor = None
plt.ion()
pipeline.start()
while True:
    inDepth = None 
    imuMsg = None
    while inDepth is None or imuMsg is None :
        if inDepth is None  :
            inDepth = stereo_queue.tryGet()
        if imuMsg is None :
            imuMsg = imuQueue.tryGet()

    q = imuMsg.packets[0].rotationVector
    quat = [q.i, q.j, q.k, q.real]

    depth = inDepth.getFrame().astype(np.float32)   # meters

    if rays_processor is None:
        h, w = depth.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        fovx = 2 * np.arctan(w / (2 * fx))
        fovy = 2 * np.arctan(h / (2 * fy))
        rays_processor = RaysProcessor(fovx, fovy, 1)

    xy = rays_processor.get_ray_points_local_frame(depth, quat)
    print(xy)
    # ---- depth view ----
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("depth", d)
    cv2.waitKey(1)
    # ---- draw XY with OpenCV ----
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)

    if xy is not None and len(xy) > 0:

        x = xy[:, 0]
        y = xy[:, 1]

        x_n = (x - 0) / (2 - 0 + 1e-6)
        y_n = (y - -2) / (2 - -2 + 1e-6)

        u = (x_n * 599).astype(np.int32)
        v = (y_n * 599).astype(np.int32)

        for i in range(len(u)):
            cv2.circle(canvas, (u[i], v[i]), 3, (0, 255, 0), -1)

    cv2.imshow("XY Rays", canvas)
    cv2.waitKey(1)
