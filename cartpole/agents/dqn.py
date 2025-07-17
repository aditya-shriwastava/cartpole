#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

import torch
import numpy as np
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from cartpole.gym import CartpoleGym
from cartpole.agent import Agent


class DQN(Agent):
    def __init__(self, model_path='./runs/dqn/latest/best_model'):
        self.model_path = model_path
        self.model = SB3_DQN.load(self.model_path)

        self.discrete_actions = np.linspace(-3.0, 3.0, 11)

    def get_cmd(
        self,
        cart_pos: float,
        cart_vel: float,
        pole_angle: float,
        pole_angular_vel: float
    ) -> float:
        obs = np.array([cart_pos, cart_vel, pole_angle, pole_angular_vel])
        action, _ = self.model.predict(obs, deterministic=True) 
        return self.discrete_actions[action]


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--training-steps', type=int, default=100000,
                       help='Number of training steps')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre trained model to continue training')
    args = parser.parse_args()

    # Create dir for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"runs/dqn/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Create symlink to latest run
    latest_link = "runs/dqn/latest"
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(timestamp, latest_link)

    gym = CartpoleGym(
        discrete_action_space = True,
        max_episode_len = 1000,
    )
    gym = Monitor(gym, save_dir)
    print("[INFO]: Training gym created.")

    if args.model_path is None:
        print("[INFO]: Creating New DQN Model.")
        model = SB3_DQN(
            'MlpPolicy',
            gym,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=1.0,
            target_update_interval=500,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            train_freq=4,
            gradient_steps=1,
            verbose=0,
            device='auto',
            policy_kwargs=dict(
                net_arch=[256,256],
                activation_fn=torch.nn.ReLU,
                optimizer_class=torch.optim.Adam
            )
        )
    else:
        print(f"[INFO]: Loding DQN Model from {args.model_path}...")
        model = SB3_DQN.load(args.model_path)
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
