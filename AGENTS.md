# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

This repo is an **incomplete extract** from a larger Koopman-operator autonomous ship control project. It contains:

- `train_v4_dict_input.py` — PyTorch training script for a Koopman V4 dict-input model (ship dynamics learning).
- `motion.cpp` / `motion.h` — C++ ROS ship control node (requires full ROS workspace + Elane/Botix SDK; **not buildable in this repo**).
- `data/` — Pre-processed `.npz` voyage datasets (train/val/test splits, 10 Hz sampling from rosbag recordings).

### Critical: missing local packages

`train_v4_dict_input.py` imports two sibling packages that are **not present** in this repository:

- `koopman` — provides `paths`, `evalkit`, `setup_repo`
- `new_v4_dict_input` — provides `HorizontalKoopmanModelV4DictInput` model and evaluation functions

The script sets `REPO_ROOT = Path(__file__).resolve().parents[1]` and expects these packages as siblings one directory up. The training script **cannot be executed end-to-end** without these packages.

### What you CAN do

- **Lint**: `flake8 --max-line-length=130 --ignore=E402,F401,E501,W503 train_v4_dict_input.py`
- **Syntax check**: `python3 -m py_compile train_v4_dict_input.py`
- **Load and inspect data**: all `.npz` files in `data/` load successfully via `np.load(path, allow_pickle=True)["datas"]`. Each segment is a dict with keys: `len`, `Pos`, `Vel`, `pqr`, `Thrusters_CMD`, `Euler`.
- **PyTorch operations**: torch, numpy, pyyaml, tensorboard, matplotlib are all installed and working.

### Data format

Each `.npz` file contains `datas`, an object array of segments. Each segment is a dict:
| Key | Shape | Description |
|-----|-------|-------------|
| `Pos` | `(2, T)` | x, y position |
| `Vel` | `(2, T)` | u, v velocities |
| `pqr` | `(1, T)` | yaw rate r |
| `Thrusters_CMD` | `(4, T)` | port throttle, port angle, starboard throttle, starboard angle |
| `Euler` | `(3, T)` | roll, pitch, yaw |
| `len` | scalar | number of timesteps T |

### C++ code

The C++ files (`motion.cpp`, `motion.h`) are reference code for a ROS-based ship control node. They depend on proprietary Elane/Botix SDK, ROS messages, Eigen, spdlog, yaml-cpp, and many custom MPC/PID libraries. They **cannot be compiled** in this environment.
