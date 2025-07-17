#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

import torch
import numpy as np
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from cartpole.gym import CartpoleGym
from cartpole.agent import Agent


class SAC(Agent):
    def __init__(self, model_path='./runs/sac/latest/best_model'):
        self.model_path = model_path
        self.model = SB3_SAC.load(self.model_path)

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
    parser = argparse.ArgumentParser(description='Train SAC agent')
    parser.add_argument('--training-steps', type=int, default=100000,
                       help='Number of training steps')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre trained model to continue training')
    args = parser.parse_args()

    # Create dir for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"runs/sac/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Create symlink to latest run
    latest_link = "runs/sac/latest"
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
        print("[INFO]: Creating New SAC Model.")
        model = SB3_SAC(
            'MlpPolicy',
            gym,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            verbose=1,
            device='auto',
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU,
                optimizer_class=torch.optim.Adam
            )
        )
    else:
        print(f"[INFO]: Loading SAC Model from {args.model_path}...")
        model = SB3_SAC.load(args.model_path)
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
