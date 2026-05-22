# =====================================================================
# RL-Actor + MPC-Shield  ::  护盾干预可观测性消息
#   topic : /control/shield_intervention   (publish @ 10 Hz)
#   用途   : 监控 RL 失控风险，看护盾的「修正强度」与「触发频率」
# =====================================================================
std_msgs/Header header

# 护盾干预残差  E_shield = ||U_safe - U_rl||^2   (归一化空间)
float32 e_shield

# 是否本拍真正生效（残差超过阈值 -> true）
bool   active

# 触发的硬约束类别位掩码 :
#   bit0 : 推力盒约束          | bit1 : 舵角盒约束
#   bit2 : 转舵速率约束        | bit3 : 推力速率约束
uint8  bound_mask

# 原始 RL 输出 (4 维，归一化 ∈ [-1, 1])
float32[4] u_rl

# 护盾修正后输出 (4 维，归一化 ∈ [-1, 1])
float32[4] u_safe

# 物理量（已乘 ACTION_SCALE）：[N, rad, N, rad]，便于事后回放
float32[4] u_safe_phys

# 推理 / QP 耗时（毫秒），用于实时性监控
float32 infer_ms
float32 qp_ms
