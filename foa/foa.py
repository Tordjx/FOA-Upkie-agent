import numpy as np
from vartools.states import ObjectPose
from fast_obstacle_avoidance.obstacle_avoider import SampledClusterAvoider, FastObstacleAvoider
import numpy as np
from vartools.states import ObjectPose
import gymnasium as gym

class MockClusterer :
    def __init__(self):
        self.points = None
    def fit(self,points):
        
        self.points = points
        self.labels_ = np.zeros(points.shape[0])#np.array([i for i in range(points.shape[0])])
        
class MinimalRobot2D:
    """Simple 2D robot with fixed LIDAR at origin and circular control radius."""

    def __init__(self, control_radius, control_point=np.array([0.0, 0.0])):
        self.pose = ObjectPose(position=np.zeros(2), orientation=0.0)

        self.control_radius = control_radius
        self.control_point = control_point  # Relative to robot base

    @property
    def rotation_matrix(self):
        theta = self.pose.orientation
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])

    def transform_to_world(self, points):
        """Transform points from robot to world frame."""
        return self.rotation_matrix @ points.T + self.pose.position[:, None]

    def transform_to_robot(self, points):
        """Transform points from world to robot frame."""
        return self.rotation_matrix.T @ (points.T - self.pose.position[:, None])
class ReactiveAvoidance:
    def __init__(self, control_radius):
        # Setup robot model
        self.robot = MinimalRobot2D(control_radius = control_radius )
        self.robot.control_radius = control_radius
        self.robot.control_point = [0, 0]

        # Avoider instance
        delta = 100 * (np.pi / 180)/30 
        self.avoider = SampledClusterAvoider(control_radius=self.robot.control_radius,clusterer = MockClusterer() ,weight_factor = 2*delta,weight_power= 2)#,cluster_params = {"eps": 2 * control_radius, "min_samples": 3, "n_jobs" : -1})
        self.modulated_velocity = np.zeros(2)

    def compute(self, reference_velocity, obstacle_points):
        """Returns modulated velocity given obstacle points and reference velocity.
        
        Parameters:
        - reference_velocity: np.array of shape (2,) – desired velocity before obstacle avoidance
        - obstacle_points: np.array of shape (N, 2) – sampled points (e.g., raycast or lidar)

        Returns:
        - modulated_velocity: np.array of shape (2,)
        """
        if obstacle_points is None or obstacle_points.shape[0]<=0 :
            print('NO POINTS !!! sending ref vel')
            return reference_velocity
        if obstacle_points.shape[1] == 3 :
            obstacle_points = obstacle_points[:,:2]
        
        self.robot.pose.orientation = 0.0  # Optional: update if orientation is relevant

        # Update avoider with obstacle points
        self.avoider.update_laserscan(obstacle_points.T, in_robot_frame=False)

        # Modulate the velocity
        try : 
            modulated_velocity = self.avoider.avoid(
                reference_velocity,
                self.robot.pose.position
            )
            self.modulated_velocity = modulated_velocity
        except : 
            print("error, something happened ")
            return reference_velocity


        return modulated_velocity
