# -*- coding: utf-8 -*-
# =============================================================================
#  RL-Actor + MPC-Shield  ::  Step 3 / 4
#  文件         : src/ship_control/scripts/train_rl_optuna.py
#  功能         :
#      1) 基于 Stable-Baselines3 PPO + data_env.ShipTrackingEnv 训练 Actor
#      2) Optuna `objective(trial)` 自动采样 PPO 超参 + 奖励权重
#      3) 在“按航次切分的验证集”上计算 4 项物理指标：
#             xte_mean / heading_err_mean / smoothness_penalty / shield_rate
#         并合成综合得分 Fitness = 100 - Σ w_i·metric_i
#      4) 每个 trial 追加写入  logs/optuna_trials_log.jsonl
#      5) Best trial 的 Actor 导出为  models/rl_policy_best.onnx
#      6) 自动绘制：
#             (a) 雷达图：first trial vs best trial 的四项归一化指标
#             (b) 折线图：验证集上 raw RL 输出 vs 仿真护盾修正后输出
#
#  作者         : RL-Shield team
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---- 第三方 ML 栈（Docker 镜像已预装） -------------------------------------------
import optuna
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# ---- 项目内（第二步交付） -----------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_env as de  # noqa: E402  pylint: disable=wrong-import-position


# =============================================================================
# 0. 全局常量 & 综合得分权重
#    （这些是 Fitness 公式中的元权重 w1..w4，**不是** 环境奖励权重）
# =============================================================================
FITNESS_W: Dict[str, float] = {
    "xte_mean":           1.0,   # 每米横向误差扣 1 分
    "heading_err_mean":   30.0,  # 每弧度航向误差扣 30 分（≈ 0.52 / 度）
    "smoothness_penalty": 5.0,   # 每单位控制总变差扣 5 分
    "shield_rate":        20.0,  # 护盾干预率每 1.0（100%）扣 20 分
}

# 仿真护盾参数（与第 4 步 C++ OSQP 护盾的硬约束保持一致）
SHIELD_RATE_LIMIT: float = math.radians(15.0)  # 转舵速率上限 15°/s
SHIELD_ABS_LIMIT:  np.ndarray = np.ones(4, dtype=np.float32)  # 归一化 |a|<=1
SHIELD_TRIGGER_EPS: float = 1e-3  # ‖U_safe - U_rl‖² > eps 视为护盾激活


# =============================================================================
# 1. 仿真版 MPC-Shield（与 C++ OSQP 同口径，用于评估 shield_rate）
#    目标：  min ||u_safe - u_rl||²  s.t.  |u_safe| <= 1
#                                          |Δu_safe| <= rate_max * dt
#    对盒约束 + 速率约束的解可写成 closed form：先逐维 clip 到全局盒，再对
#    与上一拍 a_prev 的差做速率盒约束 clip。这就是 OSQP 在该简化情形下
#    的最优解。
# =============================================================================
def sim_shield_step(
    u_rl: np.ndarray,
    a_prev: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, float]:
    """返回护盾修正后控制 u_safe 以及干预残差 e_shield = ||u_safe-u_rl||²。"""
    u_rl = np.asarray(u_rl, dtype=np.float32)
    a_prev = np.asarray(a_prev, dtype=np.float32)
    rate_max_norm = (SHIELD_RATE_LIMIT * dt) / de.MAX_RUDDER_RAD  # 舵维归一化速率上限
    # 推力维不限速率（推力控制带宽足够），舵维 (dl, dr) 限速
    rate_box = np.array([np.inf, rate_max_norm, np.inf, rate_max_norm], dtype=np.float32)

    u_safe = np.clip(u_rl, -SHIELD_ABS_LIMIT, SHIELD_ABS_LIMIT)
    u_safe = np.clip(u_safe, a_prev - rate_box, a_prev + rate_box)
    e_shield = float(np.sum((u_safe - u_rl) ** 2))
    return u_safe.astype(np.float32), e_shield


# =============================================================================
# 2. 验证集 rollout：计算 4 项物理指标
# =============================================================================
def evaluate_policy_on_val(
    model: PPO,
    val_env: de.ShipTrackingEnv,
    n_episodes: int = 5,
    record_traces: bool = False,
) -> Dict[str, Any]:
    """在验证集上 rollout n_episodes 个 episode，计算指标。"""
    xte_acc:    List[float] = []
    yaw_acc:    List[float] = []
    tv_acc:     List[float] = []   # 控制总变差 (Total Variation)
    shield_hit: List[float] = []   # 是否触发护盾（0/1）
    traces: List[Dict[str, np.ndarray]] = []

    for ep in range(n_episodes):
        obs, _info = val_env.reset(seed=10_000 + ep)
        a_prev = np.zeros(de.ACTION_DIM, dtype=np.float32)
        tv_ep: float = 0.0
        t_xte: List[float] = []
        t_yaw: List[float] = []
        u_rl_log:   List[np.ndarray] = []
        u_safe_log: List[np.ndarray] = []
        hit_log:    List[int] = []

        done = False
        while not done:
            a_rl, _ = model.predict(obs, deterministic=True)
            a_rl = np.asarray(a_rl, dtype=np.float32).reshape(-1)
            a_safe, e_shield = sim_shield_step(a_rl, a_prev, val_env.cfg.dt)
            shield_active = e_shield > SHIELD_TRIGGER_EPS
            val_env.set_shield_active(shield_active)

            # 用 a_safe（已通过护盾）推进环境，对齐部署链路
            obs, _r, term, trunc, info = val_env.step(a_safe)

            tv_ep += float(np.sum(np.abs(a_safe - a_prev)))
            t_xte.append(abs(float(info["xte"])))
            t_yaw.append(abs(float(info["heading_err"])))
            hit_log.append(int(shield_active))
            if record_traces:
                u_rl_log.append(a_rl.copy())
                u_safe_log.append(a_safe.copy())

            a_prev = a_safe
            done = bool(term or trunc)

        xte_acc.append(float(np.mean(t_xte) if t_xte else 0.0))
        yaw_acc.append(float(np.mean(t_yaw) if t_yaw else 0.0))
        tv_acc.append(tv_ep)
        shield_hit.append(float(np.mean(hit_log) if hit_log else 0.0))

        if record_traces and u_rl_log:
            traces.append(
                {
                    "u_rl":   np.stack(u_rl_log,   axis=0),
                    "u_safe": np.stack(u_safe_log, axis=0),
                }
            )

    metrics: Dict[str, Any] = {
        "xte_mean":           float(np.mean(xte_acc)),
        "heading_err_mean":   float(np.mean(yaw_acc)),
        "smoothness_penalty": float(np.mean(tv_acc)),
        "shield_rate":        float(np.mean(shield_hit)),
    }
    if record_traces:
        metrics["traces"] = traces  # 仅当显式要求时才返回，避免污染 JSONL
    return metrics


def compute_fitness(metrics: Dict[str, float]) -> float:
    return float(
        100.0
        - FITNESS_W["xte_mean"]           * metrics["xte_mean"]
        - FITNESS_W["heading_err_mean"]   * metrics["heading_err_mean"]
        - FITNESS_W["smoothness_penalty"] * metrics["smoothness_penalty"]
        - FITNESS_W["shield_rate"]        * metrics["shield_rate"]
    )


# =============================================================================
# 3. JSONL 结构化日志
# =============================================================================
def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")


# =============================================================================
# 4. PPO Actor → ONNX 导出
#    SB3 的 ActorCriticPolicy 内部含 features_extractor + mlp_extractor.policy_net
#    + action_net；为了部署侧只跑“确定性 actor”，我们 wrap 一个纯前向模块。
# =============================================================================
class _DeterministicActorWrapper(nn.Module):
    """把 SB3 PPO 策略剥成 obs(13) -> action(4) 的纯前向 net，便于 ONNX 导出。"""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.policy.extract_features(obs)
        # SB3 在 SharedNet / NonSharedNet 两种情况下 extract_features 行为不同
        if isinstance(feats, tuple):
            pi_features, _ = feats
        else:
            pi_features = feats
        latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        mean_actions = self.policy.action_net(latent_pi)
        # tanh squash → 严格落到 [-1, 1]，与 C++ 侧 ACTION_SCALE 反归一化口径一致
        return torch.tanh(mean_actions)


def export_actor_to_onnx(model: PPO, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.policy.set_training_mode(False)
    wrapper = _DeterministicActorWrapper(model.policy).to("cpu").eval()
    dummy = torch.zeros((1, de.STATE_DIM), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # 在新的 torch.onnx dynamo 导出路径上，权重默认会被外部化到一个并行
    # 的 `.onnx.data` 文件，这会让 C++ 部署侧（只读单一 .onnx）失败。
    # 这里强制把权重重新打包回 ONNX 内部，并清理临时 sidecar。
    try:
        import onnx  # noqa: WPS433  (local import 仅在导出阶段需要)
        from onnx.external_data_helper import load_external_data_for_model
        m = onnx.load(str(out_path), load_external_data=False)
        has_external = any(
            init.data_location == onnx.TensorProto.EXTERNAL
            for init in m.graph.initializer
        )
        if has_external:
            load_external_data_for_model(m, str(out_path.parent))
            for init in m.graph.initializer:
                init.ClearField("data_location")
                init.ClearField("external_data")
            onnx.save(m, str(out_path))
            sidecar = out_path.with_suffix(out_path.suffix + ".data")
            if sidecar.exists():
                sidecar.unlink()
    except Exception as exc:  # 不让重打包失败阻断训练总流程
        print(f"[onnx] WARN 重打包外部权重失败: {exc}")

    print(f"[onnx] 导出策略到  {out_path}  ({out_path.stat().st_size/1024:.1f} KiB)")


# =============================================================================
# 5. Optuna  objective
# =============================================================================
def _build_env_cfg_from_trial(trial: optuna.Trial) -> de.EnvConfig:
    """从 trial 抽样奖励权重 + 域随机化强度，落到 EnvConfig。"""
    return de.EnvConfig(
        w_xte=          trial.suggest_float("w_xte",      0.5,  5.0,  log=True),
        w_heading=      trial.suggest_float("w_heading",  1.0, 10.0,  log=True),
        w_speed=        trial.suggest_float("w_speed",    0.05, 1.0,  log=True),
        w_smooth=       trial.suggest_float("w_smooth",   0.1,  5.0,  log=True),  # 抖舵惩罚
        w_rate=         trial.suggest_float("w_rate",     0.1,  5.0,  log=True),
        w_shield_hint=  trial.suggest_float("w_shield_hint", 0.0, 1.0),
        hydro_sigma=    trial.suggest_float("hydro_sigma",   0.02, 0.08),
        obs_noise_pos=  trial.suggest_float("obs_noise_pos", 0.1,  1.0),
    )


def _build_ppo_kwargs_from_trial(trial: optuna.Trial) -> Dict[str, Any]:
    return dict(
        learning_rate= trial.suggest_float("lr",       1e-5, 5e-4, log=True),
        n_steps=       trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        batch_size=    trial.suggest_categorical("batch_size", [64, 128, 256]),
        gamma=         trial.suggest_float("gamma",    0.95, 0.999),
        gae_lambda=    trial.suggest_float("gae_lambda", 0.9, 0.99),
        clip_range=    trial.suggest_float("clip_range", 0.1, 0.3),
        ent_coef=      trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True),
        vf_coef=       trial.suggest_float("vf_coef",  0.3, 1.0),
        max_grad_norm= trial.suggest_float("max_grad_norm", 0.3, 1.0),
        n_epochs=      trial.suggest_int("n_epochs", 3, 10),
    )


def make_objective(
    train_npz: str,
    val_npz:   str,
    total_timesteps: int,
    out_dir: Path,
    jsonl_path: Path,
    eval_episodes: int,
    n_envs: int = 1,
    seed: int = 42,
) -> Callable[[optuna.Trial], float]:

    def objective(trial: optuna.Trial) -> float:
        t_start = time.time()
        env_cfg = _build_env_cfg_from_trial(trial)
        ppo_kw  = _build_ppo_kwargs_from_trial(trial)

        # ---- 训练环境（向量化） ------------------------------------------------
        thunks = [de.make_vec_env_fn(train_npz, cfg=env_cfg, train=True,
                                     seed_base=seed + 1000 * trial.number)(i)
                  for i in range(n_envs)]
        train_vec = VecMonitor(DummyVecEnv([lambda env=e: env for e in thunks]))

        # ---- PPO 训练 -------------------------------------------------------
        model = PPO(
            policy="MlpPolicy",
            env=train_vec,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            verbose=0,
            seed=seed + trial.number,
            tensorboard_log=str(out_dir / "tb"),
            device="auto",
            **ppo_kw,
        )
        try:
            model.learn(total_timesteps=total_timesteps, progress_bar=False)
        except Exception as exc:  # 训练侧崩溃直接 prune
            train_vec.close()
            raise optuna.TrialPruned(f"PPO learn failed: {exc}") from exc

        # ---- 验证集评估 -----------------------------------------------------
        val_env = de.make_env(val_npz, cfg=env_cfg, train=False, seed=seed + 7)
        metrics = evaluate_policy_on_val(model, val_env, n_episodes=eval_episodes)
        fitness = compute_fitness(metrics)
        elapsed = time.time() - t_start

        # ---- JSONL 追加 -----------------------------------------------------
        record = {
            "trial_number": int(trial.number),
            "timestamp":    time.time(),
            "elapsed_sec":  round(elapsed, 2),
            "params":       trial.params,
            "env_cfg":      asdict(env_cfg),
            "metrics":      metrics,           # xte_mean / heading / smoothness / shield_rate
            "fitness":      fitness,
        }
        append_jsonl(jsonl_path, record)

        # ---- 把策略字典临时存到磁盘，best trial 复用 ------------------------
        ckpt_path = out_dir / f"trial_{trial.number:04d}.zip"
        model.save(ckpt_path)
        trial.set_user_attr("ckpt_path", str(ckpt_path))
        trial.set_user_attr("metrics", metrics)

        train_vec.close()
        print(f"[trial {trial.number:03d}] fitness={fitness:+7.3f}  "
              f"xte={metrics['xte_mean']:.2f}m  yaw={math.degrees(metrics['heading_err_mean']):.2f}°  "
              f"smooth={metrics['smoothness_penalty']:.2f}  shield={metrics['shield_rate']:.2%}  "
              f"({elapsed:.1f}s)")
        return fitness  # Optuna 默认 maximize

    return objective


# =============================================================================
# 6. 可视化：(a) 雷达图  (b) 动作对比折线图
# =============================================================================
def plot_radar_first_vs_best(jsonl_path: Path, png_path: Path) -> None:
    """读 JSONL，画 first trial vs best trial 的 4 指标雷达图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not jsonl_path.exists():
        print(f"[viz] 找不到 {jsonl_path}，跳过雷达图")
        return
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("[viz] JSONL 为空，跳过雷达图")
        return
    first = rows[0]
    best = max(rows, key=lambda r: r["fitness"])

    keys = ["xte_mean", "heading_err_mean", "smoothness_penalty", "shield_rate"]
    labels_display = ["XTE [m]", "Heading [rad]", "Smoothness TV", "Shield rate"]
    # 雷达图需要把指标归一化到 [0,1]：越大越差 → 反向显示（1 - norm）
    arr_first = np.array([first["metrics"][k] for k in keys], dtype=np.float32)
    arr_best  = np.array([best ["metrics"][k] for k in keys], dtype=np.float32)
    denom = np.maximum(np.maximum(arr_first, arr_best), 1e-6)
    score_first = 1.0 - (arr_first / denom)   # 1.0 = perfect
    score_best  = 1.0 - (arr_best  / denom)

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    score_first = np.concatenate([score_first, score_first[:1]])
    score_best  = np.concatenate([score_best,  score_best[:1]])
    angles      = angles + angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, score_first, "o-", linewidth=2,
            label=f"First trial (#{first['trial_number']}, fit={first['fitness']:.2f})")
    ax.fill(angles, score_first, alpha=0.15)
    ax.plot(angles, score_best,  "o-", linewidth=2,
            label=f"Best  trial (#{best ['trial_number']}, fit={best ['fitness']:.2f})")
    ax.fill(angles, score_best,  alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_display)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Optuna 优化前后物理指标对比（越外越好）", pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.20, -0.10))
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[viz] 雷达图  → {png_path}")


def plot_action_overlay(
    model: PPO,
    val_env: de.ShipTrackingEnv,
    png_path: Path,
    max_steps: int = 300,
) -> None:
    """折线图：raw RL 输出 vs 仿真护盾修正输出（4 通道）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = evaluate_policy_on_val(model, val_env, n_episodes=1, record_traces=True)
    traces = metrics.get("traces", [])
    if not traces:
        print("[viz] 没有 trace，跳过动作折线图")
        return
    tr = traces[0]
    u_rl = tr["u_rl"][:max_steps]
    u_sf = tr["u_safe"][:max_steps]
    t = np.arange(u_rl.shape[0]) * val_env.cfg.dt
    ch_names = ["thrust_left", "rudder_left", "thrust_right", "rudder_right"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, ch_names)):
        ax.plot(t, u_rl[:, i], "-",  lw=1.2, alpha=0.85, label="RL raw")
        ax.plot(t, u_sf[:, i], "--", lw=1.4, alpha=0.95, label="Sim-Shield")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("验证集动作对比：RL 原始输出 vs 仿真护盾修正后", y=0.995)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[viz] 动作对比折线图 → {png_path}")


# =============================================================================
# 7. main
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PPO + Optuna 训练 / 评估 / 导出 ONNX")
    p.add_argument("--data",      type=str, default="/ws/data/koopman_train_merged.npz")
    p.add_argument("--val",       type=str, default="/ws/data/koopman_val.npz")
    p.add_argument("--out",       type=str, default="/ws/models")
    p.add_argument("--logs",      type=str, default="/ws/logs")
    p.add_argument("--n-trials",  type=int, default=30)
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--n-envs",    type=int, default=1)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--study-name", type=str, default="ship_rl_ppo")
    p.add_argument("--storage",   type=str, default=None,
                   help="optuna RDB URL（可选）；不填则使用内存存储")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir  = Path(args.out)
    logs_dir = Path(args.logs)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = logs_dir / "optuna_trials_log.jsonl"
    if jsonl_path.exists():
        # 同名 study 续跑时不要追加到旧 JSONL，避免雷达图把多次实验混在一起
        backup = jsonl_path.with_suffix(f".{int(time.time())}.bak.jsonl")
        jsonl_path.rename(backup)
        print(f"[init] 旧 JSONL 已备份为 {backup}")

    print(f"[init] device={'cuda' if torch.cuda.is_available() else 'cpu'}  "
          f"n_trials={args.n_trials}  timesteps/trial={args.timesteps}")
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    objective = make_objective(
        train_npz=args.data,
        val_npz=args.val,
        total_timesteps=args.timesteps,
        out_dir=out_dir,
        jsonl_path=jsonl_path,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # ---- 提取 best trial → 导出 ONNX -------------------------------------------
    best = study.best_trial
    print(f"\n[best] trial #{best.number}  fitness={best.value:.3f}")
    for k, v in best.params.items():
        print(f"       {k:>16s} = {v}")
    ckpt_path = Path(best.user_attrs["ckpt_path"])
    best_model = PPO.load(ckpt_path, device="cpu")
    onnx_path = out_dir / "rl_policy_best.onnx"
    export_actor_to_onnx(best_model, onnx_path)

    # ---- 把 best trial 元数据落盘，供 C++ 部署侧读取 ---------------------------
    meta = {
        "best_trial":   int(best.number),
        "fitness":      float(best.value),
        "params":       best.params,
        "metrics":      best.user_attrs.get("metrics", {}),
        "onnx_path":    str(onnx_path),
        "state_dim":    int(de.STATE_DIM),
        "action_dim":   int(de.ACTION_DIM),
        "action_scale": de.ACTION_SCALE.tolist(),
    }
    (out_dir / "rl_policy_best.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[best] 元数据  → {out_dir / 'rl_policy_best.json'}")

    # ---- 可视化 -----------------------------------------------------------
    plot_radar_first_vs_best(jsonl_path, logs_dir / "radar_first_vs_best.png")
    val_env = de.make_env(args.val, cfg=de.EnvConfig(**{
        k: v for k, v in best.params.items() if k in de.EnvConfig().__dataclass_fields__
    }), train=False, seed=args.seed + 9)
    plot_action_overlay(best_model, val_env, logs_dir / "action_overlay_best.png")

    print(f"\n[done] all artefacts under: {out_dir}  &  {logs_dir}")


if __name__ == "__main__":
    main()
