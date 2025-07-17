from cartpole.agent import Agent


class PID(Agent):
    def __init__(self):
        # PID gains for pole angle control
        self.kp_angle = 30.0  # Proportional gain for angle
        self.ki_angle = 0.11   # Integral gain for angle
        self.kd_angle = 5.0  # Derivative gain for angle

        # PID gains for cart position control
        self.kp_pos = 1.6     # Proportional gain for position
        self.ki_pos = 0.025    # Integral gain for position
        self.kd_pos = 0.8     # Derivative gain for position

        # Integral error accumulators
        self.integral_angle = 0.0
        self.integral_pos = 0.0

        # Previous measurement for derivative on measurement (prevents derivative kick)
        self.prev_angle = 0.0
        self.prev_pos = 0.0
        self.prev_angular_vel = 0.0

        # Anti-windup limits
        self.integral_limit = 10.0

        # Feed-forward gains
        self.kff_cart_vel = 0.25    # Feed-forward gain for cart velocity

        # Time step (matches simulation timestep)
        self.dt = 0.02

    def get_cmd(
        self,
        cart_pos: float,
        cart_vel: float,
        pole_angle: float,
        pole_angular_vel: float
    ) -> float:
        # Primary control: Keep pole upright (angle = 0)
        error_angle = 0.0 - pole_angle

        # Update integral term with anti-windup
        self.integral_angle += error_angle * self.dt
        self.integral_angle = max(-self.integral_limit, min(self.integral_limit, self.integral_angle))

        # Derivative on measurement instead of error (prevents derivative kick)
        angle_derivative = -(pole_angle - self.prev_angle) / self.dt

        # PID control for angle
        cmd_angle = (self.kp_angle * error_angle +
                     self.ki_angle * self.integral_angle +
                     self.kd_angle * angle_derivative)

        # Secondary control: Keep cart centered (position = 0)
        error_pos = 0.0 - cart_pos

        # Update integral term with anti-windup
        self.integral_pos += error_pos * self.dt
        self.integral_pos = max(-self.integral_limit, min(self.integral_limit, self.integral_pos))

        # Derivative on measurement for position
        pos_derivative = -(cart_pos - self.prev_pos) / self.dt

        # Feed-forward term for cart velocity
        ff_cart_vel = self.kff_cart_vel * cart_vel

        # PID + Feed-forward control for position
        cmd_pos = (self.kp_pos * error_pos +
                   self.ki_pos * self.integral_pos +
                   self.kd_pos * pos_derivative +
                   ff_cart_vel)

        # Combine both control signals
        # Angle control is primary, position control is secondary
        total_cmd = cmd_angle + cmd_pos

        # Update previous values for next iteration
        self.prev_angle = pole_angle
        self.prev_pos = cart_pos
        self.prev_angular_vel = pole_angular_vel

        # Clamp output to actuator limits [-3, 3]
        return max(-3.0, min(3.0, total_cmd))
