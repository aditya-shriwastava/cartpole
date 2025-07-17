from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_cmd(
        self,
        cart_pos: float,
        cart_vel: float,
        pole_angle: float,
        pole_angular_vel: float
    ) -> float:
        pass
