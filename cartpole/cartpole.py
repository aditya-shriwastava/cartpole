#!/usr/bin/env python3
import time
import argparse

import mujoco
import mujoco.viewer
from cartpole.agents.pid import PID


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
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("cartpole.xml")
        self.data = mujoco.MjData(self.model)
        self.control_hz = 1/self.model.opt.timestep
        self.reward = 0;

    def cmd(self, cmd):
        self.data.ctrl[0] = cmd

    def get_observation(self):
        cart_pos = self.data.qpos[0]
        pole_angle = -self.data.qpos[1]
        cart_vel = self.data.qvel[0]
        pole_angular_vel = -self.data.qvel[1]
        return cart_pos, pole_angle, cart_vel, pole_angular_vel

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
                if abs(pole_angle) > 0.2:
                    print(f"|pose_angle| > 0.2, terminating. Total reward collected: {self.reward}")
                    break
                else:
                    self.reward += 1

                action = agent.get_cmd(cart_pos, pole_angle, cart_vel, pole_angular_vel)
                self.cmd(action)

                mujoco.mj_step(self.model, self.data)

                print(f"Cart pos: {cart_pos:.3f}, Cart vel: {cart_vel:.3f}, Pole angle: {pole_angle:.3f}, Pole angular vel: {pole_angular_vel:.3f}")

                viewer.sync()
                rate.sleep()


def main():
    parser = argparse.ArgumentParser(description='CartPole control')
    parser.add_argument('--agent', type=str, required=True,
                       choices=['PID', 'MPC', 'DQN', 'PPO', 'SAC'],
                       help='Control agent to use')
    args = parser.parse_args()

    # Instantiate agent
    if args.agent == 'PID':
        agent = PID()
    elif args.agent == 'MPC':
        raise NotImplementedError("MPC agent not implemented yet")
    elif args.agent == 'DQN':
        raise NotImplementedError("DQN agent not implemented yet")
    elif args.agent == 'PPO':
        raise NotImplementedError("PPO agent not implemented yet")
    elif args.agent == 'SAC':
        raise NotImplementedError("SAC agent not implemented yet")

    cartpole = Cartpole()
    cartpole.control_loop(agent)


if __name__ == '__main__':
    main()
