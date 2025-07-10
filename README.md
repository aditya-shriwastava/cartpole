# cartpole
> Can we balance the pole by applying forces in the left and right to the cart

## Getting Started
1. **Clone the repository**
```bash
git clone https://github.com/aditya-shriwastava/cartpole.git
cd cartpole
```

2. **Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install .
```

4. **Usage**
After installation, you can run cartpole with:
```bash
cartpole
```

## Controls to implement
* [ ] PID (Simplest Control, Just works in many cases)
* [ ] MPC (Receding horizon optimal control, Go to classical control approach for most robotics problems)
* [ ] DQN (Simple, Sample Efficient, First approach to show good results on atari games)
* [ ] PPO (Go to approach for most of the robotics control problem in simulator, Wall clock time efficient)
* [ ] SAC (Sample Efficient)
