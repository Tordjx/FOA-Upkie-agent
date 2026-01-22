import zmq
from upkie.utils.raspi import configure_agent_process, on_raspi

if on_raspi():
    configure_agent_process()
    reg_freq = False
else:
    reg_freq = False

from tqdm import tqdm
from foa.foa import ReactiveAvoidance
import numpy as np
from env.envs import make_rays_pink_env
from loop_rate_limiters import RateLimiter
import gymnasium as gym
import torch
import upkie.envs
import gin
from config.settings import EnvSettings
from upkie.utils.robot_state import RobotState

upkie.envs.register()

gin.parse_config_file(f"config/settings.gin")
env_settings = EnvSettings()

gym.envs.registration.register(
    id="UpkieServos-v5", entry_point="env.upkie_servos:UpkieServos"
)

reactive_avoidance = ReactiveAvoidance(control_radius=0.3)
import logging
import time
import os
def setup_logger(method_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("logs", f"{method_name}_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any old handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(filename, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return filename

def modulate_velocity(reactive_avoidance, i):
    target_forward = -i["spine_observation"]["joystick"]["left_axis"][1]
    target_yaw = -i["spine_observation"]["joystick"]["left_axis"][0]
    reference_velocity = np.array([target_forward, target_yaw])
    obstacle_points = i["obstacle_points"]
    modulated_velocity = reactive_avoidance.compute(reference_velocity, obstacle_points)
    modulated_velocity[1] *= -1
    modulated_velocity = np.linalg.inv(np.diag([1, 0.10])) @ modulated_velocity  # upkie's lever arm
    correction = modulated_velocity - reference_velocity
    print(modulated_velocity, obstacle_points)
    return correction

agent_frequency = env_settings.agent_frequency
max_episode_duration = 25000
spine_config = env_settings.spine_config
spine_config["base_orientation"]  = {'rotation_base_to_imu':np.array([1,0,0,0,-1,0,0,0,-1],dtype=float)}
velocity_env = gym.make(
    env_settings.env_id,
    max_episode_steps=int(max_episode_duration * agent_frequency),
    frequency=100,
    regulate_frequency=reg_freq,
    shm_name="upkie",
    spine_config=spine_config,
    fall_pitch=np.pi / 2,
    init_state=RobotState(position_base_in_world=np.array([2, 2, 0.58]))
)

env = make_rays_pink_env(
    velocity_env,
    env_settings,
    eval_mode=False,
)

def main():
    #log_file = setup_logger("foa")
    #logging.info(f"Starting run, logging to {log_file}")
    # ZeroMQ publisher setup
    #context = zmq.Context()
    #socket = context.socket(zmq.PUB)
    #socket.bind("tcp://*:8080")  # Change port if needed

    rate_limiter = RateLimiter(frequency=10)
    s, i = env.reset()

    for _ in tqdm(range(200000)):
        rate_limiter.sleep()
        
        # Send obstacle points
        obstacle_points = i["obstacle_points"]
        #socket.send_pyobj(obstacle_points)  # Publish numpy array (pickled)

        action = modulate_velocity(reactive_avoidance, i)
        
        s, r, d, t, i = env.step(action)
        if i["spine_observation"]["joystick"]["triangle_button"] or d:
            obs, i = env.reset()
        joystick_input = i["spine_observation"]["joystick"]['left_axis']
        forward_velocity = i["spine_observation"]["wheel_odometry"]["velocity"]
        yaw_velocity = i["spine_observation"]["base_orientation"][
            "angular_velocity"
        ][2]
        """
        logging.info(
            f"Action={action}, Joystick={joystick_input}, rdot={forward_velocity}, phidot={yaw_velocity} "
            f"ObstaclePoints={obstacle_points}"
        )"""

if __name__ == "__main__":
    main()
