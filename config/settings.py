#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

from dataclasses import dataclass
from typing import List, Optional, Tuple

import gin


@gin.configurable
@dataclass
class EnvSettings:
    """!
    Environment settings.
    """
    agent_frequency: int
    env_id: str
    spine_config: dict
    spine_frequency: int
    width: int
    height: int
    image_every: int
    max_episode_duration: float
    window: bool

