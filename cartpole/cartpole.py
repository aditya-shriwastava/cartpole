#!/usr/bin/env python3
import math
import time
import argparse
import os
from typing import Optional, List

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from cartpole.agents.pid import PID
from cartpole.agents.mpc import MPC


class Rate:
    def __init__(self, hz: float):
        self.__hz = hz
        self.__time = self.now()

    def now(self):
        return time.monotonic()

    def sleep(self):
        sleep_duration = max(
            0,
            (1 / self.__hz) - (self.now() - self.__time)
        )
        time.sleep(sleep_duration)
        self.__time = self.now()


class Cartpole:
    def __init__(self, create_gif: bool = False, verbose: bool = False):
        self.create_gif = create_gif
        self.verbose = verbose

        self.model = mujoco.MjModel.from_xml_path("cartpole.xml")
        self.data = mujoco.MjData(self.model)

        self.control_hz = 1/self.model.opt.timestep
        self.max_frames = 100  # Maximum number of frames to capture
        self.frame_capture_interval = 5  # Capture frame every N control cycles

        self.reward = 0
        self.frames: List[np.ndarray] = []
        self.control_cycle_count = 0

    def cmd(self, cmd):
        self.data.ctrl[0] = cmd

    def get_observation(self):
        cart_pos = self.data.qpos[0]
        pole_angle = -self.data.qpos[1]
        cart_vel = self.data.qvel[0]
        pole_angular_vel = -self.data.qvel[1]
        return cart_pos, pole_angle, cart_vel, pole_angular_vel

    def capture_frame(self, viewer):
        """Capture a frame from the viewer for GIF creation"""
        if self.create_gif:
            # Capture frame every N control cycles
            if self.control_cycle_count % self.frame_capture_interval == 0:
                # Check if we haven't exceeded max frames limit
                if len(self.frames) < self.max_frames:
                    # Use dimensions within MuJoCo's default framebuffer limits
                    width, height = 640, 480

                    # Create renderer and render frame
                    renderer = mujoco.Renderer(self.model, height=height, width=width)
                    renderer.update_scene(self.data)
                    pixels = renderer.render()

                    # Convert to PIL Image format (RGB)
                    self.frames.append(pixels)

    def save_gif(self, filename: str = "cartpole_episode.gif"):
        """Save captured frames as a GIF"""
        if not self.create_gif or not self.frames:
            return

        print(f"Saving GIF with {len(self.frames)} frames to {filename} ...")

        # Convert numpy arrays to PIL Images
        pil_frames = []
        for frame in self.frames:
            pil_frames.append(Image.fromarray(frame.astype(np.uint8)))

        # Calculate frame duration in milliseconds based on control interval
        # Each frame represents frame_capture_interval control cycles
        effective_fps = self.control_hz / self.frame_capture_interval
        frame_duration = int(1000 / effective_fps)

        # Save as GIF
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration,
            loop=0  # Infinite loop
        )
        print(f"Done")

    def control_loop(self, agent):
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:
            rate = Rate(self.control_hz)
            while viewer.is_running():
                cart_pos, pole_angle, cart_vel, pole_angular_vel = self.get_observation()
                if abs(pole_angle) > math.pi/4:
                    print(f"|pose_angle| > pi/4, terminating!")
                    viewer.close()
                    break
                elif abs(pole_angle) <= 0.2:
                    self.reward += 1

                cmd = agent.get_cmd(cart_pos, pole_angle, cart_vel, pole_angular_vel)
                self.cmd(cmd)

                mujoco.mj_step(self.model, self.data)

                if self.verbose:
                    print(f"Cart pos: {cart_pos:.3f}, Cart vel: {cart_vel:.3f}, Pole angle: {pole_angle:.3f}, Pole angular vel: {pole_angular_vel:.3f}, Force applied: {cmd:.3f}")

                # Increment control cycle counter
                self.control_cycle_count += 1

                # Capture frame for GIF if enabled
                self.capture_frame(viewer)

                viewer.sync()
                rate.sleep()

            print(f"Total reward collected: {self.reward}")
            # Save GIF if recording was enabled
            if self.create_gif:
                self.save_gif()


def main():
    parser = argparse.ArgumentParser(description='CartPole control')
    parser.add_argument('--agent', type=str, required=True,
                       choices=['PID', 'MPC', 'MPC_ROBUST', 'DQN', 'PPO', 'SAC'],
                       help='Control agent to use')
    parser.add_argument('--create-gif', action='store_true',
                       help='Record episode as GIF (max 10 seconds)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output during simulation')
    args = parser.parse_args()

    # Instantiate agent
    if args.agent == 'PID':
        agent = PID()
    elif args.agent == 'MPC':
        agent = MPC()
    elif args.agent == 'DQN':
        raise NotImplementedError("DQN agent not implemented yet")
    elif args.agent == 'PPO':
        raise NotImplementedError("PPO agent not implemented yet")
    elif args.agent == 'SAC':
        raise NotImplementedError("SAC agent not implemented yet")

    cartpole = Cartpole(create_gif=args.create_gif, verbose=args.verbose)
    cartpole.control_loop(agent)


if __name__ == '__main__':
    main()
