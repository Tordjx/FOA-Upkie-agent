import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms import functional as F

from config.settings import EnvSettings

env_settings = EnvSettings()


class ImageAugmentation(gym.Wrapper):
    def __init__(
        self,
        env,
        image_every=1,
        eval_mode=False,
        contrast=0.5,
        brightness=0.3,
        hue=0.4,
        saturation=0.5,
        noise_std=0.01,
    ):
        super().__init__(env)
        self.image_every = image_every
        self.image_count = 0
        self.eval_mode = eval_mode
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.noise_std = noise_std

    def transform_generator(self):
        brightness = np.random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast = np.random.uniform(1 - self.contrast, 1 + self.contrast)
        hue = np.random.uniform(-self.hue, self.hue)
        saturation = np.random.uniform(1 - self.saturation, 1.5 + self.saturation)
        np.random.uniform(0, self.noise_std)

        def episode_transform(x):
            x = F.adjust_brightness(x, brightness)
            x = F.adjust_contrast(x, contrast)
            x = F.adjust_hue(x, hue)
            x = F.adjust_saturation(x, saturation)
            # Keep values in valid range
            return x

        def noiser(x):
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0, 1)
            return x

        return episode_transform, noiser

    def reset(self, **kwargs):
        self.image_count = 0
        self.episode_transform, self.noiser = self.transform_generator()
        s, i = self.env.reset(**kwargs)
        image = i["image"]
        image = self.episode_transform(torch.from_numpy(image)).numpy()
        i["clean_image"] = image
        image = self.noiser(torch.from_numpy(image)).numpy()
        self.image = image
        i["image"] = image
        return s, i

    def step(self, action):
        s, r, d, t, i = self.env.step(action)

        if self.image_count % self.image_every == 0:
            image = i["image"]
            clean_image = self.episode_transform(torch.from_numpy(image)).numpy()
            i["clean_image"] = clean_image
            image = self.noiser(torch.from_numpy(clean_image)).numpy()
            self.image = image
            self.clean_image = clean_image
        else:
            clean_image = self.clean_image
            image = self.image
        i["image"] = image
        i["clean_image"] = clean_image
        self.image_count += 1
        return s, r, d, t, i
