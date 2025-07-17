import casadi as ca
import numpy as np

from cartpole.agent import Agent
from cartpole.agents.pid import PID


class MPC(Agent):
    def __init__(self):
        # Constants
        self.m = 10.0 # Mass of pole
        self.M = 10.0 # Mass of cart
        self.g = 9.81 # Acceleration due to gravity
        self.l = 0.3 # Half length of the pole
        self.gear_ratio = 100

        # MPC parameters
        self.N = 50 # Horizon
        self.dt = 0.02 # Timestep
        self.Q = ca.diag([60, 0, 40, 0]) # Penalize State Error
        self.R = ca.diag([0]) # Penalize Control Efforts

        # Store previous solution for warm starting
        self.prev_U = None
        self.prev_X = None

        self.f = self.get_continuous_dynamics()
        self.opti, self.U, self.X, self.X0 = self.setup_mpc()

        self.pid = PID()

    def get_continuous_dynamics(self):
        r"""
        Cart pole dynamics:
        $$\ddot{x}cos(\theta) + gsin(\theta)l = \frac{4}{3}l\ddot{\theta}$$
        $$(M+m)\ddot{x} = F + ml\dot{\theta}^2sin(\theta)-ml\ddot{\theta}cos(\theta)$$

        Where,
        x: position of cart
        $\theta$: pole angle ccw +ve
        m: mass of pole
        M: mass of cart
        l: half length of pole
        g: acceleration due to gravity
        F: applied force on cart
        """
        # State variables
        x = ca.SX.sym('x')
        dx = ca.SX.sym('dx')
        th = ca.SX.sym('th')
        dth = ca.SX.sym('dth')

        # Control variables
        F = ca.SX.sym('F')

        # Equations of motion
        num_fp = F + self.m * self.l * dth**2 * ca.sin(th)
        num_sp = self.m * self.g * ca.sin(th) * ca.cos(th)
        num = (4/3) * num_fp + num_sp
        den = (self.m * ca.cos(th)**2) + ((4/3) * (self.M + self.m))
        ddx = num / den
        ddth = (ddx * ca.cos(th) + self.g * ca.sin(th)) / ((4/3) * self.l)

        f = ca.Function(
            'f',
            [ca.vertcat(x, dx, th, dth), F],
            [ca.vertcat(dx, ddx, dth, ddth)]
        )
        return f

    def euler_integration(self, f, x_k, u_k, dt):
        x_kp1 = x_k + dt * f(x_k, u_k)
        return x_kp1

    def rk4_integration(self, f, x_k, u_k, dt):
        k1 = f(x_k, u_k)
        k2 = f(x_k + 0.5 * dt * k1, u_k)
        k3 = f(x_k + 0.5 * dt * k2, u_k)
        k4 = f(x_k + dt * k3, u_k)
        x_kp1 = x_k + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_kp1

    def setup_mpc(self):
        opti = ca.Opti()

        X = opti.variable(4, self.N+1) # Predicted state trajectory
        U = opti.variable(1, self.N) # Control inputs
        X0 = opti.parameter(4) # Initial state

        cost = 0.0
        for k in range(self.N):
            opti.subject_to(X[:,k+1] == self.rk4_integration(self.f, X[:,k], U[:,k], self.dt))
            cost += ca.mtimes([X[:,k].T, self.Q, X[:,k]]) + ca.mtimes([U[:,k].T, self.R, U[:,k]])

        opti.subject_to(X[:,0] == X0)
        opti.minimize(cost)

        opti.subject_to(opti.bounded(-300, U, 300))

        opti.solver(
            'ipopt',
            {
                "ipopt.max_cpu_time": 0.1,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.sb": "yes"
            }
        )

        return opti, U, X, X0

    def get_cmd(
        self,
        cart_pos: float,
        cart_vel: float,
        pole_angle: float,
        pole_angular_vel: float
    ) -> float:
        try:
            # Set current state as initial condition
            current_state = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
            self.opti.set_value(self.X0, current_state)

            # Warm start with previous solution
            if self.prev_U is not None and self.prev_X is not None:
                # Convert to numpy for easier manipulation
                U_prev = np.array(self.prev_U)
                X_prev = np.array(self.prev_X)

                # Shift control sequence: U[0:N-1] <- U[1:N], U[N-1] <- U[N-1]
                U_shifted = np.append(U_prev.flatten()[1:], U_prev.flatten()[-1])

                # Shift state trajectory: X[:,0:N] <- X[:,1:N+1], X[:,N] <- X[:,N]
                X_shifted = np.hstack([X_prev[:, 1:], X_prev[:, -1:]])

                # Convert back to CasADi format and set initial guess
                self.opti.set_initial(self.U, U_shifted.reshape(1, -1))
                self.opti.set_initial(self.X, X_shifted)
            else:
                # Cold start: use zeros for initial guess
                self.opti.set_initial(self.U, ca.DM.zeros(1, self.N))
                self.opti.set_initial(self.X, ca.DM.zeros(4, self.N+1))

            # Solve the optimization problem
            sol = self.opti.solve()

            # Store solution for next warm start
            self.prev_U = sol.value(self.U)
            self.prev_X = sol.value(self.X)

            # Debug: Check shapes
            # print(f"U shape: {self.prev_U.shape}, X shape: {self.prev_X.shape}")

            # Extract the first control action and convert to motor command
            U_array = np.array(self.prev_U)
            optimal_control = U_array.flatten()[0]
            return float(optimal_control)/self.gear_ratio

        except Exception as e:
            # Fallback to PID if optimization fails
            print(f"MPC optimization failed: {e}. Switching to PID")
            return self.pid.get_cmd(cart_pos, pole_angle, cart_vel, pole_angular_vel)
