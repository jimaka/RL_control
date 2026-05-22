# RL-Actor + MPC-Shield  ::  65 m Bulker Autonomous Control

End-to-end pipeline for a 65-meter bulk carrier:

1. **Python** : domain-randomised Gymnasium env  +  PPO trained with **Optuna**
   sweep on physical KPIs (xte, heading, smoothness, shield-rate) and exported
   to **ONNX**.
2. **C++ / ROS** : the same ONNX policy is consumed by a real-time node whose
   output is filtered by an **OSQP MPC-Shield**; the shield residual is
   published on `/control/shield_intervention` for live observability.

## Project tree

```
.
├── data/                                  # ← .npz datasets  (mounted read-only)
│   ├── koopman_train_merged.npz
│   ├── koopman_val.npz
│   └── ...
├── docker/
│   ├── Dockerfile                         # CUDA 11.8 + ROS Noetic + ORT/OSQP
│   └── docker-compose.yml                 # services: trainer / shield_node / shell
├── models/                                # ← rl_policy_best.onnx lands here
├── logs/                                  # ← optuna_trials_log.jsonl, tb runs, plots
├── motion.cpp                             # legacy reference (existing controller)
├── motion.h
└── src/
    └── ship_control/                      # ROS catkin package
        ├── CMakeLists.txt                 # built in step 4
        ├── package.xml                    # built in step 4
        ├── config/                        # YAML : shield bounds, observation spec
        ├── include/
        │   └── ship_control/
        │       └── motion_rl_shield.hpp   # step 4
        ├── launch/
        │   └── motion_rl_shield.launch    # step 4
        ├── msg/
        │   └── ShieldIntervention.msg     # step 4
        ├── scripts/                       # Python side
        │   ├── data_env.py                # step 2 (domain randomisation env)
        │   └── train_rl_optuna.py         # step 3 (Optuna loop + viz)
        └── src/
            └── motion_rl_shield.cpp       # step 4 (ORT + OSQP shield ROS node)
```

## Quick start (Docker)

```bash
# 1) Build the image (CUDA + ROS + ORT + OSQP + Python ML stack)
docker compose -f docker/docker-compose.yml build

# 2) Train  →  produces models/rl_policy_best.onnx + logs/optuna_trials_log.jsonl + plots
docker compose -f docker/docker-compose.yml run --rm trainer

# 3) Launch the ROS shield node  →  consumes the ONNX model
docker compose -f docker/docker-compose.yml run --rm shield_node

# 4) Interactive shell (debug / Optuna dashboard / Jupyter)
docker compose -f docker/docker-compose.yml run --rm shell
```

GPU passthrough is enabled (`deploy.resources … nvidia`). The local `data/`,
`models/`, `logs/` folders are bind-mounted, so artefacts persist across runs.
