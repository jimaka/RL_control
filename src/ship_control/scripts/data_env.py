# -*- coding: utf-8 -*-
# =============================================================================
#  RL-Actor + MPC-Shield  ::  Step 2 / 4
#  文件         : src/ship_control/scripts/data_env.py
#  功能         :
#      1) 解析 koopman_*.npz 数据集（按 voyage 的 dict 结构存储）
#      2) 严格按“航次”切分训练 / 验证集，禁止 Random Split
#      3) 把绝对坐标的预测目标，严格变换到 t 时刻随体坐标系下
#      4) 构造 Gymnasium 环境 `ShipTrackingEnv`，在 reset() / step() 中注入
#         域随机化（hydrodynamic ±5% Gaussian、观测噪声、初始状态扰动）
#      5) 提供工厂函数 `make_env(...)` 给 train_rl_optuna.py 直接调用
#
#  状态(13 维) :
#      [Δx_b, Δy_b, Δψ,           # 前瞻航点在随体坐标系下的相对位姿误差
#       u,   v,   r,              # 随体坐标系下的纵向/横向速度、艏摇角速度
#       u_dot, v_dot, r_dot,      # 体坐标系下的加速度（数值差分）
#       tl,  dl,  tr,  dr]        # 上一拍真实施加给船的控制量（已归一化）
#
#  控制(4 维)  :
#      [thrust_left, rudder_left, thrust_right, rudder_right]    ∈ [-1, +1]
#      通过 ACTION_SCALE 反归一化到物理量
#
#  作者         : RL-Shield team
# =============================================================================
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 gymnasium>=0.29，请检查 Docker 镜像或 pip install gymnasium") from e


# =============================================================================
# 0. 全局常量
# =============================================================================
STATE_DIM: int = 13      # 与任务规范一致：13 列系统状态
ACTION_DIM: int = 4      # 与任务规范一致：4 列控制（左推/左舵/右推/右舵）

# 65 m 散货船的物理边界（与 motion.cpp 中的阻力/满舵约束一致）
SHIP_LENGTH_M: float = 65.0
MAX_THRUST_N: float = 60_000.0           # 单侧最大推力 [N]，等比缩放到 [-1,1]
MAX_RUDDER_RAD: float = math.radians(90) # 最大满舵 ±90°（与 C++ 护盾保持一致）
MAX_RUDDER_RATE: float = math.radians(15)  # 转舵速率上限 15°/s（用作环境侧惩罚）

# 控制量归一化系数：a_norm ∈ [-1,1]   →   a_phys = a_norm * ACTION_SCALE
ACTION_SCALE: np.ndarray = np.array(
    [MAX_THRUST_N, MAX_RUDDER_RAD, MAX_THRUST_N, MAX_RUDDER_RAD],
    dtype=np.float32,
)

# 数据采样周期：dataset 名字暗示 10 Hz；与 motion.cpp 的控制周期一致
DEFAULT_DT: float = 0.1


# =============================================================================
# 1. 通用几何工具：随体坐标系 (body frame) 变换
# =============================================================================
def world_to_body(dx_world: np.ndarray, dy_world: np.ndarray, yaw_t: float) -> Tuple[np.ndarray, np.ndarray]:
    """把世界系下的相对位移 (dx, dy) 旋转到 t 时刻随体坐标系。

    Args:
        dx_world : (N,) 世界系 x 偏移 = x_target - x_t
        dy_world : (N,) 世界系 y 偏移 = y_target - y_t
        yaw_t    : t 时刻船的艏向角 [rad]   （右手系，z 轴向上）

    Returns:
        dx_body, dy_body : 随体系下的纵向 / 横向偏移
    """
    c, s = math.cos(yaw_t), math.sin(yaw_t)
    dx_body = c * dx_world + s * dy_world
    dy_body = -s * dx_world + c * dy_world
    return dx_body, dy_body


def wrap_to_pi(angle: float) -> float:
    """把角度归一化到 (-π, π]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# =============================================================================
# 2. 数据加载与航次切分
# =============================================================================
@dataclass
class Voyage:
    """单条航次的标准化容器。所有量均按时间序列存储，长度 T_i 可不同。"""
    pos_xy:    np.ndarray  # (T, 2)   世界系位置
    vel_uv:    np.ndarray  # (T, 2)   随体系速度 (surge, sway)
    yaw:       np.ndarray  # (T,)     艏向角 (rad)，由 Euler[2] 取得
    yaw_rate:  np.ndarray  # (T,)     艏摇角速度 r
    ctrl:      np.ndarray  # (T, 4)   控制量 [tl, dl, tr, dr]
    dt:        float = DEFAULT_DT

    @property
    def length(self) -> int:
        return int(self.pos_xy.shape[0])


def load_voyages(npz_path: str) -> List[Voyage]:
    """从 `koopman_*.npz` 读取航次列表。

    每个文件结构：`{'datas': np.ndarray(dtype=object)}`，元素为 dict，键含
    Pos(2,T) / Vel(2,T) / pqr(1,T) / Thrusters_CMD(4,T) / Euler(3,T) / len。
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"数据集不存在: {npz_path}")
    raw = np.load(npz_path, allow_pickle=True)
    if "datas" not in raw:
        raise KeyError(f"{npz_path} 中缺少 'datas' 键")
    voyages: List[Voyage] = []
    for item in raw["datas"]:
        d: Dict[str, Any] = item  # type: ignore[assignment]
        pos = np.asarray(d["Pos"], dtype=np.float32).T            # (T, 2)
        vel = np.asarray(d["Vel"], dtype=np.float32).T            # (T, 2)
        pqr = np.asarray(d["pqr"], dtype=np.float32).reshape(-1)  # (T,)
        eul = np.asarray(d["Euler"], dtype=np.float32).T          # (T, 3)
        ctl = np.asarray(d["Thrusters_CMD"], dtype=np.float32).T  # (T, 4)
        T = min(pos.shape[0], vel.shape[0], pqr.shape[0], eul.shape[0], ctl.shape[0])
        if T < 16:
            continue  # 段太短，跳过
        voyages.append(
            Voyage(
                pos_xy=pos[:T],
                vel_uv=vel[:T],
                yaw=eul[:T, 2],
                yaw_rate=pqr[:T],
                ctrl=ctl[:T],
            )
        )
    if not voyages:
        raise RuntimeError(f"{npz_path} 解析后没有可用航次")
    return voyages


def split_by_voyage(
    voyages: Sequence[Voyage],
    train_ratio: float = 0.8,
) -> Tuple[List[Voyage], List[Voyage]]:
    """按时间连续的航次顺序切分，严禁 Random Split。

    采用“前 N 条航次为训练，后续为验证”的策略，确保两集合在
    时间维度上彼此独立，避免 leakage。
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须落在 (0,1) 内")
    n = len(voyages)
    n_train = max(1, int(round(n * train_ratio)))
    return list(voyages[:n_train]), list(voyages[n_train:])


# =============================================================================
# 3. 简化 3-DOF 水动力代理模型（Sim-to-Real 训练用）
# =============================================================================
@dataclass
class HydroParams:
    """65 m 散货船的标称水动力参数；reset() 时会被域随机化扰动。"""
    mass:   float = 1.85e6     # 排水质量 [kg]
    iz:     float = 5.20e8     # 艏摇转动惯量 [kg·m²]
    x_u:    float = -2.10e4    # 纵向线性阻力 [N·s/m]
    y_v:    float = -8.00e4    # 横向线性阻力 [N·s/m]
    n_r:    float = -1.50e7    # 艏摇阻力 [N·m·s]
    rud_y:  float = 1.10e4     # 单度舵角产生的横向力 [N/rad]
    rud_n:  float = 3.20e5     # 单度舵角产生的艏摇力矩 [N·m/rad]
    half_b: float = 7.0        # 半船宽，左右双桨力臂 [m]


def perturb_hydro(base: HydroParams, sigma: float, rng: np.random.Generator) -> HydroParams:
    """对所有水动力参数注入 ±sigma 的高斯噪声（任务规范要求 ±5%）。"""
    def jitter(x: float) -> float:
        return float(x * (1.0 + rng.normal(0.0, sigma)))
    return HydroParams(
        mass=jitter(base.mass),
        iz=jitter(base.iz),
        x_u=jitter(base.x_u),
        y_v=jitter(base.y_v),
        n_r=jitter(base.n_r),
        rud_y=jitter(base.rud_y),
        rud_n=jitter(base.rud_n),
        half_b=base.half_b,
    )


# =============================================================================
# 4. Gymnasium 环境
# =============================================================================
@dataclass
class EnvConfig:
    """所有可调超参数。`train_rl_optuna.py` 通过 trial.suggest_* 灌入。"""
    dt:                float = DEFAULT_DT
    max_episode_steps: int   = 600
    lookahead_steps:   int   = 20         # 前瞻航点在数据集中的步长间隔
    # 域随机化强度
    hydro_sigma:       float = 0.05       # 水动力 ±5%
    init_pos_sigma:    float = 3.0        # 初始位置噪声 [m]
    init_yaw_sigma:    float = math.radians(3.0)
    obs_noise_pos:     float = 0.5        # 观测噪声：位置 [m]
    obs_noise_yaw:     float = math.radians(0.5)
    obs_noise_vel:     float = 0.05       # 速度噪声 [m/s]
    obs_noise_rate:    float = math.radians(0.3)
    # 奖励权重
    w_xte:             float = 1.0
    w_heading:         float = 2.0
    w_speed:           float = 0.2
    w_smooth:          float = 0.5        # 控制平顺性
    w_rate:            float = 1.0        # 转舵速率
    w_shield_hint:     float = 0.1        # 进入护盾边界的“软围栏”惩罚
    # 终止条件
    xte_terminate_m:   float = 25.0       # |xte| 超过即判负
    yaw_terminate_rad: float = math.radians(60.0)


class ShipTrackingEnv(gym.Env):
    """65 m 散货船航迹跟踪环境。

    每个 episode 从一条 voyage 中随机抽取一段，把记录的位置序列当作参考航迹，
    船舶动力学使用 3-DOF 代理模型在每个 step() 内积分一次。`reset()` 时对
    水动力参数与初始状态注入高斯噪声；`step()` 时对返回观测注入测量噪声，
    确保 PPO 策略获得足够的 Domain Randomization 来抗实船扰动。

    Notes
    -----
    * 严禁使用 voyage 内部的下一拍真值作为下一拍观测（那相当于 cheating）。
    * 参考航迹只用于：① 计算 xte、heading 误差；② 提供前瞻航点用于观测。
    """

    metadata = {"render_modes": []}

    # ---------------------------------------------------------------------
    # 构造 / 配置
    # ---------------------------------------------------------------------
    def __init__(
        self,
        voyages: Sequence[Voyage],
        cfg: Optional[EnvConfig] = None,
        hydro_base: Optional[HydroParams] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not voyages:
            raise ValueError("voyages 不能为空")
        self._voyages: List[Voyage] = list(voyages)
        self.cfg: EnvConfig = cfg or EnvConfig()
        self._hydro_base: HydroParams = hydro_base or HydroParams()
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        # 运行时状态（reset 中初始化）
        self._voyage: Optional[Voyage] = None
        self._hydro: HydroParams = self._hydro_base
        self._t_idx: int = 0                  # 在当前 voyage 中的“参考航点指针”
        self._step_count: int = 0
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0
        self._u: float = 0.0
        self._v: float = 0.0
        self._r: float = 0.0
        self._u_dot: float = 0.0
        self._v_dot: float = 0.0
        self._r_dot: float = 0.0
        self._a_prev: np.ndarray = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_shield_active: bool = False  # 由外部（train 脚本/部署）回填

    # ---------------------------------------------------------------------
    # gymnasium API : reset
    # ---------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._voyage = self._rng.choice(self._voyages)  # type: ignore[assignment]
        v = self._voyage
        # 随机选一个起点，并保证后面还有 max_episode_steps + lookahead 这么长
        max_start = max(1, v.length - self.cfg.max_episode_steps - self.cfg.lookahead_steps - 2)
        self._t_idx = int(self._rng.integers(0, max_start))
        self._step_count = 0

        # 用航次起点的真值 + 高斯扰动 当作船的初始状态  → 域随机化①
        self._x = float(v.pos_xy[self._t_idx, 0] + self._rng.normal(0, self.cfg.init_pos_sigma))
        self._y = float(v.pos_xy[self._t_idx, 1] + self._rng.normal(0, self.cfg.init_pos_sigma))
        self._yaw = wrap_to_pi(
            float(v.yaw[self._t_idx] + self._rng.normal(0, self.cfg.init_yaw_sigma))
        )
        self._u = float(v.vel_uv[self._t_idx, 0])
        self._v = float(v.vel_uv[self._t_idx, 1])
        self._r = float(v.yaw_rate[self._t_idx])
        self._u_dot = self._v_dot = self._r_dot = 0.0
        self._a_prev = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_shield_active = False

        # 水动力 ±5% 高斯扰动  → 域随机化②
        self._hydro = perturb_hydro(self._hydro_base, self.cfg.hydro_sigma, self._rng)

        return self._build_obs(), self._build_info()

    # ---------------------------------------------------------------------
    # gymnasium API : step
    # ---------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a_norm = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        a_phys = a_norm * ACTION_SCALE

        # 1) 一阶 3-DOF 水动力积分
        self._integrate(a_phys, self.cfg.dt)

        # 2) 推进参考航点
        self._t_idx += 1
        self._step_count += 1

        # 3) 计算奖励与终止
        xte, heading_err, ref_speed = self._tracking_errors()
        reward = self._compute_reward(a_norm, xte, heading_err, ref_speed)
        terminated = self._is_terminated(xte, heading_err)
        truncated = (
            self._step_count >= self.cfg.max_episode_steps
            or self._t_idx >= self._voyage.length - self.cfg.lookahead_steps - 2  # type: ignore[union-attr]
        )

        self._a_prev = a_norm.copy()
        info = self._build_info(extra={
            "xte": float(xte),
            "heading_err": float(heading_err),
            "ref_speed": float(ref_speed),
            "action_norm": a_norm.tolist(),
            "shield_active": bool(self._last_shield_active),
        })
        return self._build_obs(), float(reward), bool(terminated), bool(truncated), info

    # ---------------------------------------------------------------------
    # 物理模拟：3-DOF 简化 Nomoto + 推力差分
    # ---------------------------------------------------------------------
    def _integrate(self, a_phys: np.ndarray, dt: float) -> None:
        tl, dl, tr, dr = a_phys  # 推力 [N]，舵角 [rad]
        h = self._hydro

        # 体坐标系下合外力：左右桨推力 + 左右舵的横向力 + 线性水阻力
        f_x = (tl + tr) + h.x_u * self._u
        f_y = h.rud_y * (dl + dr) + h.y_v * self._v
        # 艏摇力矩：左右桨推力差产生的转矩 + 左右舵力矩 + 艏摇阻力
        m_z = (tr - tl) * h.half_b + h.rud_n * (dl + dr) + h.n_r * self._r

        u_dot = f_x / h.mass + self._r * self._v
        v_dot = f_y / h.mass - self._r * self._u
        r_dot = m_z / h.iz

        # 半隐式 Euler
        self._u += u_dot * dt
        self._v += v_dot * dt
        self._r += r_dot * dt
        # 体速度回投到世界系
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        self._x += (c * self._u - s * self._v) * dt
        self._y += (s * self._u + c * self._v) * dt
        self._yaw = wrap_to_pi(self._yaw + self._r * dt)
        self._u_dot, self._v_dot, self._r_dot = u_dot, v_dot, r_dot

    # ---------------------------------------------------------------------
    # 跟踪误差（必须把绝对参考轨迹变到 t 时刻随体坐标系！）
    # ---------------------------------------------------------------------
    def _tracking_errors(self) -> Tuple[float, float, float]:
        v = self._voyage
        t = min(self._t_idx, v.length - 1)  # type: ignore[union-attr]
        ref_x = float(v.pos_xy[t, 0]); ref_y = float(v.pos_xy[t, 1])  # type: ignore[union-attr]
        ref_yaw = float(v.yaw[t])                                     # type: ignore[union-attr]
        ref_speed = math.hypot(float(v.vel_uv[t, 0]), float(v.vel_uv[t, 1]))  # type: ignore[union-attr]

        dx_world = ref_x - self._x
        dy_world = ref_y - self._y
        _, dy_body = world_to_body(np.array([dx_world]), np.array([dy_world]), self._yaw)
        xte = float(dy_body[0])  # 横向误差 = 体系 y 方向偏移
        heading_err = wrap_to_pi(ref_yaw - self._yaw)
        return xte, heading_err, ref_speed

    # ---------------------------------------------------------------------
    # 奖励函数：xte / heading / speed / 平顺性 / 转舵速率
    # ---------------------------------------------------------------------
    def _compute_reward(
        self,
        a_norm: np.ndarray,
        xte: float,
        heading_err: float,
        ref_speed: float,
    ) -> float:
        c = self.cfg
        # 主要任务项
        r_xte = -c.w_xte * (xte * xte)
        r_yaw = -c.w_heading * (heading_err * heading_err)
        r_spd = -c.w_speed * (math.hypot(self._u, self._v) - ref_speed) ** 2

        # 平顺性：相邻拍控制量差 → 抑制高频抖舵
        d_a = a_norm - self._a_prev
        r_smooth = -c.w_smooth * float(np.sum(d_a * d_a))

        # 转舵速率（仅看 dl/dr 两维）：超出 MAX_RUDDER_RATE 则线性扣分
        rud_rate = np.abs(d_a[[1, 3]]) * (MAX_RUDDER_RAD / c.dt)
        over = np.clip(rud_rate - MAX_RUDDER_RATE, 0.0, None)
        r_rate = -c.w_rate * float(np.sum(over))

        # 软围栏：靠近终止边界时提前给负反馈，缓解 sparse-reward
        soft_xte = max(0.0, abs(xte) - 0.6 * c.xte_terminate_m)
        r_soft = -c.w_shield_hint * (soft_xte * soft_xte)

        return float(r_xte + r_yaw + r_spd + r_smooth + r_rate + r_soft)

    def _is_terminated(self, xte: float, heading_err: float) -> bool:
        return abs(xte) > self.cfg.xte_terminate_m or abs(heading_err) > self.cfg.yaw_terminate_rad

    # ---------------------------------------------------------------------
    # 观测：13 维。前瞻航点必须变到 t 时刻随体坐标系
    # ---------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        v = self._voyage
        t_look = min(self._t_idx + self.cfg.lookahead_steps, v.length - 1)  # type: ignore[union-attr]
        ref_x = float(v.pos_xy[t_look, 0])   # type: ignore[union-attr]
        ref_y = float(v.pos_xy[t_look, 1])   # type: ignore[union-attr]
        ref_yaw = float(v.yaw[t_look])       # type: ignore[union-attr]

        dx_body, dy_body = world_to_body(
            np.array([ref_x - self._x]),
            np.array([ref_y - self._y]),
            self._yaw,
        )
        d_yaw = wrap_to_pi(ref_yaw - self._yaw)

        obs = np.array(
            [
                float(dx_body[0]),  # Δx_b
                float(dy_body[0]),  # Δy_b
                float(d_yaw),       # Δψ
                self._u, self._v, self._r,
                self._u_dot, self._v_dot, self._r_dot,
                float(self._a_prev[0]), float(self._a_prev[1]),
                float(self._a_prev[2]), float(self._a_prev[3]),
            ],
            dtype=np.float32,
        )

        # 域随机化③：观测噪声（模拟 IMU / GNSS / 推算器测量误差）
        noise = np.array(
            [
                self._rng.normal(0, self.cfg.obs_noise_pos),
                self._rng.normal(0, self.cfg.obs_noise_pos),
                self._rng.normal(0, self.cfg.obs_noise_yaw),
                self._rng.normal(0, self.cfg.obs_noise_vel),
                self._rng.normal(0, self.cfg.obs_noise_vel),
                self._rng.normal(0, self.cfg.obs_noise_rate),
                self._rng.normal(0, self.cfg.obs_noise_vel),
                self._rng.normal(0, self.cfg.obs_noise_vel),
                self._rng.normal(0, self.cfg.obs_noise_rate),
                0.0, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        return obs + noise

    # ---------------------------------------------------------------------
    # info 辅助
    # ---------------------------------------------------------------------
    def _build_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "t_idx": int(self._t_idx),
            "voyage_len": int(self._voyage.length) if self._voyage is not None else 0,
            "pose_world": (self._x, self._y, self._yaw),
            "body_vel": (self._u, self._v, self._r),
        }
        if extra:
            info.update(extra)
        return info

    # ---------------------------------------------------------------------
    # 给外部（train_rl_optuna.py 的“仿真护盾”）回填护盾状态用
    # ---------------------------------------------------------------------
    def set_shield_active(self, active: bool) -> None:
        self._last_shield_active = bool(active)


# =============================================================================
# 5. 工厂函数：供 train_rl_optuna.py 使用
# =============================================================================
def make_env(
    npz_path: str,
    cfg: Optional[EnvConfig] = None,
    train: bool = True,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> ShipTrackingEnv:
    """单一入口：加载 npz → 按航次切分 → 构造 ShipTrackingEnv。"""
    voyages = load_voyages(npz_path)
    train_v, val_v = split_by_voyage(voyages, train_ratio=train_ratio)
    chosen = train_v if train else val_v if val_v else train_v
    return ShipTrackingEnv(voyages=chosen, cfg=cfg, seed=seed)


def make_vec_env_fn(
    npz_path: str,
    cfg: Optional[EnvConfig] = None,
    train: bool = True,
    train_ratio: float = 0.8,
    seed_base: int = 0,
):
    """返回一个无参可调用对象，给 SB3 的 DummyVecEnv / SubprocVecEnv 用。"""
    def _thunk(rank: int = 0):
        return make_env(npz_path, cfg=cfg, train=train, train_ratio=train_ratio, seed=seed_base + rank)
    return _thunk


# =============================================================================
# 6. 自检入口（直接执行可冒烟测试）
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser("ship_tracking_env smoke test")
    parser.add_argument("--data", type=str, default="/ws/data/koopman_train_merged.npz")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    env = make_env(args.data, train=True, seed=42)
    obs, info = env.reset()
    print(f"[smoke] obs.shape={obs.shape}  action.shape={env.action_space.shape}")
    print(f"[smoke] voyage_len={info['voyage_len']}  start_idx={info['t_idx']}")

    total_r = 0.0
    for k in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            print(f"[smoke] 提前终止 @ step={k}  term={term} trunc={trunc}")
            break
    print(f"[smoke] {k+1} 步累计奖励 = {total_r:.3f}")
