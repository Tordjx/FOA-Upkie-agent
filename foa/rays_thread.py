import numpy as np
import depthai as dai
import open3d as o3d
import threading
from loop_rate_limiters import RateLimiter


class RaysThread:
    def __init__(self, fps=10, max_points=30, threshold=-0.2, voxel_size=100.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_points = max_points
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.lock = threading.Lock()
        self.latest_points = np.zeros((0,3))
        self.running = False
        self.pitch = 0.0

        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

        self.thread = threading.Thread(target=self.run)
        self.rate_limiter = RateLimiter(frequency=fps)

    def _create_pipeline(self):
        pipeline = dai.Pipeline()
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        pointcloud = pipeline.create(dai.node.PointCloud)
        sync = pipeline.create(dai.node.Sync)
        xOut = pipeline.create(dai.node.XLinkOut)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setCamera("left")
        monoLeft.setFps(30)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setCamera("right")
        monoRight.setFps(30)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setExtendedDisparity(True)
        depth.setSubpixel(True)
        depth.setConfidenceThreshold(20)  
        #Disparities with confidence value under this threshold are accepted. Higher confidence threshold means disparities with less confidence are accepted too

        config = depth.initialConfig.get()
        config.postProcessing.speckleFilter.enable = False
        config.postProcessing.speckleFilter.speckleRange = 28
        config.postProcessing.temporalFilter.enable = False
        config.postProcessing.spatialFilter.enable = False
        config.postProcessing.thresholdFilter.minRange = 50
        config.postProcessing.thresholdFilter.maxRange = 20000
        config.postProcessing.decimationFilter.decimationFactor = 1
        depth.initialConfig.set(config)
        
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.depth.link(pointcloud.inputDepth)

        pointcloud.outputPointCloud.link(sync.inputs["pcl"])
        xOut.setStreamName("out")
        sync.out.link(xOut.input)
        xOut.input.setBlocking(False)

        return pipeline

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            inMessage = self.queue.tryGet()
            while inMessage is None :
                inMessage = self.queue.tryGet()
            pclData = inMessage["pcl"]
            points = pclData.getPoints().astype(np.float32)
    
            if points is not None and len(points) > 0:
                mask = ((points[:, 0] != 0) | (points[:, 1] != 0) | (points[:, 2] != 0)) & ((points**2).sum(1) <= (2e3)**2)
                points = points[mask]/1e3

                #print(points.shape)
                transformed = self._transform_to_robot_frame(points, self.pitch)
                #print(transformed.shape)
                filtered = transformed[transformed[:, 2] > self.threshold]
                #print(filtered.shape)
                downsampled = self.representative_points(filtered)#self.simple_subsample(filtered)#self._downsample_points(points)
                print(downsampled.shape)
                if len(downsampled.shape) ==2:
                    downsampled[:,1] *= 1
                with self.lock:
                    self.latest_points = downsampled
                """
                
                
                downsampled = self._downsample_points(points)
                transformed = self._transform_to_robot_frame(points, self.pitch)
                filtered = transformed[transformed[:, 2] > self.threshold]
                print(points.shape, downsampled.shape, filtered.shape)
                with self.lock:
                    self.latest_points = filtered"""
            
            self.rate_limiter.sleep()
    def simple_subsample(self, points):
        points = points[~np.all(points==0, 1)]/1e3
        if len(points) == 0:
            return points
        return points[np.random.choice(points.shape[0], size =30)]
    
    def representative_points(self, points, num_points=20, quantile=0.05):
        angles = np.arctan2(points[:, 1], points[:, 0])  # -pi to pi
        distances = np.linalg.norm(points[:, :2], axis=1)
        if len(angles)==0:
            return np.array([])
        bins = np.linspace(angles.min(), angles.max(), num_points + 1)
        chosen_points = []

        for i in range(num_points):
            mask = (angles >= bins[i]) & (angles < bins[i+1])
            if np.any(mask):
                dists = distances[mask]
                pts = points[mask]

                # take quantile on distances
                idx = np.argsort(dists)[int(quantile * (len(dists)-1))]
                chosen_points.append(pts[idx])

        return np.array(chosen_points)

    def o3d_filter_downsample(self, points, nb_neighbors=20, std_ratio=0.1, output_points= 20, first_downsample = 7000):
        """Remove outliers using statistical outlier removal."""
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if len(point_cloud.points)//first_downsample ==0 or len(point_cloud.points) <= 3000 :
            point_cloud = point_cloud
        else :
            point_cloud = point_cloud.uniform_down_sample(every_k_points=len(point_cloud.points)//first_downsample)
        clean_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)#point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        if len(clean_cloud.points)//output_points ==0 :
            uni_down_pcd = clean_cloud
        else :
            uni_down_pcd = clean_cloud.uniform_down_sample(every_k_points=len(clean_cloud.points)//output_points)

        return np.asarray(uni_down_pcd.points)
    def set_pitch(self, pitch_rad):
        self.pitch = pitch_rad

    def get_latest(self):
        with self.lock:
            return self.latest_points

    def _transform_to_robot_frame(self, points_pc: np.ndarray, pitch_rad: float) -> np.ndarray:
        if points_pc is None or len(points_pc) == 0:
            return np.empty((0, 3), dtype=np.float64)

        points_swapped = np.empty_like(points_pc)
        points_swapped[:, 0] = points_pc[:, 2]
        points_swapped[:, 1] = -points_pc[:, 1]
        points_swapped[:, 2] = points_pc[:, 0]

        c = np.cos(pitch_rad)
        s = np.sin(pitch_rad)
        R_y = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

        return points_swapped @ R_y.T

    import numpy as np

    def _downsample_points(self,points, az_bins=20, el_bins=3):
        """
        points: Nx3 array in robot/camera frame
            x forward, y left, z up (adapt if needed)
        az_bins: number of bins in azimuth (-pi to pi)
        el_bins: number of bins in elevation (-pi/2 to pi/2)
        """
        fx, fy= 69 * (np.pi / 180),54 * (np.pi / 180)
        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            return pts

        # Compute azimuth (θ) and elevation (φ)
        az = np.arctan2(pts[:, 1], pts[:, 0])  # -pi..pi
        el = np.arctan2(pts[:, 2], np.linalg.norm(pts[:, :2], axis=1))  # -pi/2..pi/2

        # Bin edges
        az_edges = np.linspace(-fx, fx, az_bins + 1)
        el_edges = np.linspace(-fy, fy, el_bins + 1)

        selected = []
        for i in range(az_bins):
            az_mask = (az >= az_edges[i]) & (az < az_edges[i+1])
            if not np.any(az_mask):
                continue
            for j in range(el_bins):
                mask = az_mask & (el >= el_edges[j]) & (el < el_edges[j+1])
                if not np.any(mask):
                    continue
                cell_pts = pts[mask]
                # Pick closest point in range
                idx = np.argmin(np.linalg.norm(cell_pts, axis=1))
                selected.append(cell_pts[idx])

        out = np.array(selected)/1e3
        return out[~np.all(out == 0, 1)]
