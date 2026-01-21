import gymnasium as gym
from foa.rays_thread import RaysThread

class RaysRaspiWrapper(gym.Wrapper):
    def __init__(self, env, image_every=10):
        super().__init__(env)
        self.rays_thread = RaysThread(fps=10)
        self.rays_thread.start()
        self.obstacle_points = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pitch = info["spine_observation"]["base_orientation"]["pitch"]
        self.rays_thread.set_pitch(pitch)
        self.obstacle_points = self.rays_thread.get_latest()
        info['obstacle_points'] = self.obstacle_points 
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        pitch = info["spine_observation"]["base_orientation"]["pitch"]
        self.rays_thread.set_pitch(pitch)
        self.obstacle_points = self.rays_thread.get_latest()

        info["obstacle_points"] = self.obstacle_points
        return obs, reward, done, truncated, info

    def close(self):
        self.rays_thread.stop()
        self.env.close()
