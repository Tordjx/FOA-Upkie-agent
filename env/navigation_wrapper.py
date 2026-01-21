import gin
import gymnasium as gym
import numpy as np
from pink_balancer import WholeBodyController
class NavigationWrapper(gym.Wrapper):
    """
    This wrapper makes it so that the action space is the two joystick axis,
    the observation space is the user joystick, the robot speed and the image features.
    To do so, we need to encapsulate the pink balancer
    The navigator policy runs at 10Hz
    The pink balancer runs at 100Hz
    """

    def __init__(self, env):
        super(NavigationWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(low=np.array([-1,-2]), high=np.array([0,2]), shape=(2,))
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,))
        self.dt = 1 / 100
        # Pink balancer
        gin.parse_config_file("config/pink_config.gin")
        self.controller = WholeBodyController(visualize=False)
        self.observation = None
        self.joystick = None

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        self.action = np.zeros(2)
        self.joystick = i["spine_observation"]["joystick"]["left_axis"]
        self.timestep = 0
        self.observation = i
        s = self.get_obs(i)
        return s, i
    
    def step(self, a):
        self.action = a
        for i in range(10):
            
            # print(a)
            self.observation["spine_observation"]["joystick"]["left_axis"][1] = np.clip(
                -a[0] + self.joystick[1], -1, 1
            ) 
            self.observation["spine_observation"]["joystick"]["left_axis"][1] += 0.20 #joystick hack
            self.observation["spine_observation"]["joystick"]["right_axis"][0] = (
                np.clip(a[1] + self.joystick[0], -2, 2)
            )
            action = self.controller.cycle(
                self.observation["spine_observation"], self.dt
            )["servo"]
            s, r, d, t, observation = self.env.step(action)
            self.observation = observation
            self.joystick = observation["spine_observation"]["joystick"]["left_axis"]
            
            if d or t:
                break
        s = self.get_obs(observation)
        return s, r, d, t, self.observation

    def get_obs(self, info):
        # velocity, joystick
        # features is concatenated by features wrapperi

        # forward_velocity =0.05*(info['spine_observation']['servo']['left_wheel']['velocity'] + info['spine_observation']['servo']['right_wheel']['velocity'])/(2)
        forward_velocity = info["spine_observation"]["wheel_odometry"]["velocity"]
        # print(forward_velocity, info['spine_observation']['servo']['left_wheel']['velocity'])
        yaw_velocity = info["spine_observation"]["base_orientation"][
            "angular_velocity"
        ][2]
        joystick = self.joystick
        yaw_velocity = np.clip(yaw_velocity, -1, 1)
        forward_velocity = np.clip(forward_velocity, -1.5, 1.5)
        obs = np.array([forward_velocity, yaw_velocity, joystick[0], joystick[1],self.action[0],self.action[1]])

        return obs
