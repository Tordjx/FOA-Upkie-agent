#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

import gymnasium
import numpy as np
from upkie.envs import UpkieGroundVelocity
from upkie.utils.raspi import on_raspi

from config.settings import EnvSettings
from env.navigation_wrapper import NavigationWrapper


def make_vision_pink_env(
    velocity_env: UpkieGroundVelocity,
    env_settings: EnvSettings,
    eval_mode: bool = False,
) -> gymnasium.Wrapper:
    velocity_env = NavigationWrapper(velocity_env)
    if not on_raspi():
        from env.upkie_vision import UpkieVisionWrapper

        rescaled_accel_env = UpkieVisionWrapper(
            velocity_env,
            fovx=69 * np.pi / 180,
            fovy=54 * np.pi / 180,
            model_path="data",
            height=env_settings.height,
            width=env_settings.width,
            image_every=env_settings.image_every,
            window=env_settings.window,
        )
        from env.image_augmentation_wrapper import ImageAugmentation

        rescaled_accel_env = ImageAugmentation(
            rescaled_accel_env,
            image_every=env_settings.image_every,
            eval_mode=eval_mode,
        )
        from env.features_stacker import FeaturesStackerWrapper

        rescaled_accel_env = FeaturesStackerWrapper(
            rescaled_accel_env, env_settings=env_settings
        )
    else:
        from env.raspi_vision import RaspiImageWrapper

        rescaled_accel_env = RaspiImageWrapper(
            velocity_env, image_every=env_settings.image_every
        )

    return rescaled_accel_env
