from upkie.utils.robot_state_randomization import RobotStateRandomization
from scipy.spatial.transform import Rotation as ScipyRotation
import numpy as np


class RobotStateRandomization(RobotStateRandomization):
    def __init__(self):
        super().__init__()

    def sample_position(self, np_random: np.random.Generator) -> np.ndarray:
        r"""!
        Sample a position within the given bounds.

        \param[in] np_random NumPy random number generator.
        \return Sampled position vector.
        """
        if np.random.choice([0, 1]):
            return np_random.uniform(
                low=np.array([-1.7, -3.8, 0.0]),
                high=np.array([2.8, 6.4, self.z]),
                size=3,
            )
        else:
            return np_random.uniform(
                low=np.array([2.8, -3.8, 0.0]),
                high=np.array([12.1, 1.3, self.z]),
                size=3,
            )

    def sample_orientation(self, np_random: np.random.Generator) -> np.ndarray:
        yaw = np.random.uniform(-np.pi, np.pi)
        # print(yaw)
        return ScipyRotation.from_euler("ZYX", np.array([yaw, 0.0, 0.0]))
