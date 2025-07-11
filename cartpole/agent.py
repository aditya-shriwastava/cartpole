from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_cmd(
        self,
        cart_pos: float,
        pole_angle: float,
        cart_vel: float,
        pole_angular_vel: float
    ) -> float:
        pass
