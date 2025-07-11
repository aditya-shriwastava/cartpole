#!/usr/bin/env python3
import time
import argparse

import mujoco
import mujoco.viewer


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


def main():
    parser = argparse.ArgumentParser(description='CartPole simulation')
    model = mujoco.MjModel.from_xml_path("cartpole.xml")
    data = mujoco.MjData(model)
    control_hz = 1/model.opt.timestep

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False
    ) as viewer:
        rate = Rate(control_hz)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            
            slider_pos = data.qpos[0]
            hinge_angle = -data.qpos[1]
            print(f"Slider position: {slider_pos:.3f}, Pole angle: {hinge_angle:.3f}")
            
            viewer.sync()
            rate.sleep()


if __name__ == '__main__':
    main()
