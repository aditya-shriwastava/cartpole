import math

import mujoco
import numpy as np
import gymnasium as gym


class CartpoleGym(gym.Env):
    def __init__(self, discrete_action_space = False, max_episode_len = 1000):
        super().__init__()

        self.discrete_action_space = discrete_action_space
        self.max_episode_len = max_episode_len

        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -10.0, -math.pi/2, -4 * math.pi]), 
            high=np.array([1.0, 10.0, math.pi/2, 4 * math.pi]),
            dtype=np.float64
        )

        if self.discrete_action_space:
            self.action_space = gym.spaces.Discrete(11)
            self.discrete_actions = np.linspace(-3.0, 3.0, 11)
        else:
            self.action_space = gym.spaces.Box(
                low=-3.0, 
                high=3.0,
                shape=(1,),
                dtype=np.float64
            )

        self.model = mujoco.MjModel.from_xml_path("cartpole.xml")
        self.data = mujoco.MjData(self.model)

        self.current_episode_len = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.current_episode_len = 0

        return self.get_obs(), {}

    def step(self, action):
        if self.discrete_action_space:
            action = self.discrete_actions[action]

        self.data.ctrl[0] = action
        mujoco.mj_step(self.model, self.data)
        obs = self.get_obs()
        reward = self.get_reward(obs)
        terminated = self.is_terminated(obs)

        self.current_episode_len += 1
        truncated = self.current_episode_len >= self.max_episode_len

        return np.array(obs), reward, terminated, truncated, {}

    def get_obs(self):
        cart_pos = self.data.qpos[0]
        cart_vel = self.data.qvel[0]
        pole_angle = -self.data.qpos[1]
        pole_angular_vel = -self.data.qvel[1]
        return cart_pos, cart_vel, pole_angle, pole_angular_vel

    def get_reward(self, obs):
        reward = 0.0
        cart_pos, cart_vel, pole_angle, _ = obs
        if abs(pole_angle) <= 0.2:
            reward += 1.0
            if abs(cart_pos) <= 0.2:
                reward += 1.0

        velocity_penalty = -0.02 * abs(cart_vel)
        reward += velocity_penalty

        return reward

    def is_terminated(self, obs):
        _, _, pole_angle, _ = obs

        if abs(pole_angle) > math.pi / 4:
            return True

        return False
