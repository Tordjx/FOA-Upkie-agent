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

mode = "encoder"
def make_rays_pink_env(
    velocity_env: UpkieGroundVelocity,
    env_settings: EnvSettings,
    eval_mode: bool = False,
) -> gymnasium.Wrapper:
    velocity_env = NavigationWrapper(velocity_env)
    if not on_raspi():
        from foa.rays_sim import RaysSimWrapper

        rescaled_accel_env = RaysSimWrapper(
            velocity_env,
            image_every=env_settings.image_every
        )

    else:
        print('Using rays raspi')
        if mode =="stereo":
            from foa.rays_raspi import RaysRaspiWrapper
            rescaled_accel_env = RaysRaspiWrapper(
                velocity_env,
                image_every=env_settings.image_every
            )
        elif mode =="encoder":
            from foa.monocular_raspi import MonocularWrapper
            rescaled_accel_env = MonocularWrapper(velocity_env)
        else :
            raise Exception('Supported modes are stereo or encoder')
    return rescaled_accel_env
