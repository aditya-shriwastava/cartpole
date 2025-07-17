#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

import torch
import numpy as np
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from cartpole.gym import CartpoleGym
from cartpole.agent import Agent


class PPO(Agent):
    def __init__(self, model_path='./runs/ppo/latest/best_model'):
        self.model_path = model_path
        self.model = SB3_PPO.load(self.model_path)

    def get_cmd(
        self,
        cart_pos: float,
        cart_vel: float,
        pole_angle: float,
        pole_angular_vel: float
    ) -> float:
        obs = np.array([cart_pos, cart_vel, pole_angle, pole_angular_vel])
        action, _ = self.model.predict(obs, deterministic=True)
        return np.clip(action[0], -3.0, 3.0)


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--training-steps', type=int, default=100000,
                       help='Number of training steps')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre trained model to continue training')
    args = parser.parse_args()

    # Create dir for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"runs/ppo/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Create symlink to latest run
    latest_link = "runs/ppo/latest"
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(timestamp, latest_link)

    gym = CartpoleGym(
        discrete_action_space=False,
        max_episode_len=1000,
    )
    gym = Monitor(gym, save_dir)
    print("[INFO]: Training gym created.")

    if args.model_path is None:
        print("[INFO]: Creating New PPO Model.")
        model = SB3_PPO(
            'MlpPolicy',
            gym,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device='auto',
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU,
                optimizer_class=torch.optim.Adam
            )
        )
    else:
        print(f"[INFO]: Loading PPO Model from {args.model_path}...")
        model = SB3_PPO.load(args.model_path)
    print(f"[INFO]: Done With Model Creation.")

    eval_callback = EvalCallback(
        gym,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )

    print("[INFO]: Starting Learning Process...")
    model.learn(total_timesteps=args.training_steps, callback=eval_callback)
    print("[INFO]: Done Learning.")

    print(f"[INFO]: Saving Model to {save_dir}...")
    model.save(os.path.join(save_dir, "model"))
    print("[INFO]: Done Saving Model.")


if __name__ == '__main__':
    main()
