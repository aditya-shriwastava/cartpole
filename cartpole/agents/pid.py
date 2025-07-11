from cartpole.agent import Agent

class PID(Agent):
    def __init__(self):
        pass

    def get_cmd(
        self,
        cart_pos: float,
        pole_angle: float,
        cart_vel: float,
        pole_angular_vel: float
    ) -> float:
        return 0.0
