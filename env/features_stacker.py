from collections import deque

import gymnasium as gym
import numpy as np
import torch

from autoencoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeaturesStackerWrapper(gym.Wrapper):
    def __init__(self, env, env_settings, feature_size=32):
        super().__init__(env)
        self.env_settings = env_settings
        n = feature_size
        low = np.concatenate([env.observation_space.low, -10 * np.ones(n)])
        self.observation_space = gym.spaces.Box(
            low=low,
            high=np.concatenate([env.observation_space.high, 10 * np.ones(n)]),
            shape=low.shape,
            dtype=env.observation_space.dtype,
        )
        self.features_count = 0
        self.features_memory = deque(maxlen=1)
        self.encoder = AutoEncoder(
            input_shape=(3, env_settings.height, env_settings.width),
            z_size=feature_size,
        ).to(device)
        self.encoder.load_state_dict(torch.load("autoencoder.pth", map_location=device))
        self.encoder.eval()

    def reset(self, **kwargs):
        self.features_count = 0
        s, i = self.env.reset(**kwargs)
        # Initialize memory to full of the first features
        with torch.no_grad():
            self.features_memory.append(
                self.encoder.encode(
                    torch.from_numpy(i["image"].copy()).to(device).to(torch.float32)
                )
                .cpu()
                .numpy()
            )
        self.featuress = np.concatenate(self.features_memory).reshape(-1)
        s = np.concatenate((s, self.featuress))
        return s, i

    def step(self, action):
        s, r, d, t, i = self.env.step(action)

        if self.features_count % self.env_settings.image_every == 0:
            # Pop oldest features and add the most recent
            with torch.no_grad():
                self.features_memory.append(
                    self.encoder.encode(
                        torch.from_numpy(i["image"].copy()).to(device).to(torch.float32)
                    )
                    .cpu()
                    .numpy()
                )
            self.featuress = np.concatenate(self.features_memory).reshape(-1)
        self.features_count += 1
        s = np.concatenate((s, self.featuress))
        return s, r, d, t, i
