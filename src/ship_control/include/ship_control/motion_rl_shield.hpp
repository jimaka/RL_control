// =============================================================================
//  RL-Actor + MPC-Shield  ::  Step 4 / 4
//  文件        : src/ship_control/include/ship_control/motion_rl_shield.hpp
//  功能        :
//      1) Ort::Session 加载 PPO 导出的 rl_policy_best.onnx
//      2) Eigen 预分配缓冲实现 obs(13) → action(4) 零拷贝推理
//      3) OSQP-Eigen 求解 MPC-Shield QP :
//             min  || U_safe - U_rl ||²  +  w_smooth · || ΔU_safe ||²
//             s.t. |U_safe| ≤ 1     (推力盒约束 + 满舵盒约束)
//                  |ΔU_safe| ≤ rate_max · dt   (转舵速率 + 推力速率)
//      4) /control/shield_intervention 话题广播 E_shield = ||U_safe-U_rl||²
//
//  作者        : RL-Shield team
// =============================================================================
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

#include <ship_control/ShieldIntervention.h>
#include <elane_msgs/ControlCmd.h>

namespace elane {
namespace control {

// 与 Python 训练侧 STATE_DIM / ACTION_DIM 严格一致
inline constexpr int kStateDim  = 13;
inline constexpr int kActionDim = 4;

// ---------------------------------------------------------------------------
//  护盾约束 / 权重打包
// ---------------------------------------------------------------------------
struct ShieldConfig {
  double control_dt = 0.1;

  std::array<double, kActionDim> action_scale{
      {60000.0, 1.5707963267948966, 60000.0, 1.5707963267948966}};
  std::array<double, kActionDim> abs_lo{{-1, -1, -1, -1}};
  std::array<double, kActionDim> abs_hi{{ 1,  1,  1,  1}};

  // 物理速率上限
  double rudder_rate_max_rad_s = 0.2617993878;  // 15 deg/s
  double thrust_rate_max_n_s   = 80000.0;

  // QP 权重
  double w_dev    = 1.0;   //  || U_safe - U_rl ||²
  double w_smooth = 0.5;   //  || ΔU_safe ||²

  // 干预报警
  double trigger_eps = 1e-3;
  double warn_rate   = 0.30;
  int    warn_window = 100;
};

// ---------------------------------------------------------------------------
//  ORT 推理 wrapper（零拷贝）
// ---------------------------------------------------------------------------
class OnnxActor {
 public:
  OnnxActor(const std::string& model_path, bool use_cuda);

  // 输入 obs[kStateDim]，输出 action[kActionDim]（均为归一化 ∈ [-1,1]）
  // 返回 ms 级耗时，供 /control/shield_intervention 可观测
  double Infer(const Eigen::Matrix<float, kStateDim, 1>& obs,
               Eigen::Matrix<float, kActionDim, 1>* action);

 private:
  Ort::Env env_;
  Ort::SessionOptions opts_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo mem_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  // 预分配的输入 / 输出 buffer（Eigen 持有，ORT 用 raw 指针包装为 Tensor）
  Eigen::Matrix<float, kStateDim, 1>  obs_buf_  = Eigen::Matrix<float, kStateDim,  1>::Zero();
  Eigen::Matrix<float, kActionDim, 1> act_buf_  = Eigen::Matrix<float, kActionDim, 1>::Zero();

  std::vector<int64_t> in_shape_{1, kStateDim};
  std::vector<int64_t> out_shape_{1, kActionDim};
  std::string in_name_str_, out_name_str_;
  const char* in_names_[1]{nullptr};
  const char* out_names_[1]{nullptr};
};

// ---------------------------------------------------------------------------
//  MPC-Shield (OSQP)
// ---------------------------------------------------------------------------
class MpcShield {
 public:
  explicit MpcShield(const ShieldConfig& cfg);

  // u_rl   : 归一化 RL 动作（输入）
  // u_prev : 上一拍真实施加的归一化动作（用于速率约束 + 平顺项）
  // u_safe : 修正后归一化动作（输出）
  // 返回值 : QP 求解耗时 [ms]；同时把激活的硬约束类别 bitmask 写到 bound_mask
  double Solve(const Eigen::Matrix<float, kActionDim, 1>& u_rl,
               const Eigen::Matrix<float, kActionDim, 1>& u_prev,
               Eigen::Matrix<float, kActionDim, 1>* u_safe,
               uint8_t* bound_mask);

  const ShieldConfig& cfg() const { return cfg_; }

 private:
  void BuildStaticMatrices();      // P 矩阵 + 常量约束 A
  void RefreshDynamic(const Eigen::Matrix<float, kActionDim, 1>& u_rl,
                      const Eigen::Matrix<float, kActionDim, 1>& u_prev);
  uint8_t ComputeBoundMask(const Eigen::Vector4d& u_safe,
                           const Eigen::Vector4d& u_prev) const;

  ShieldConfig cfg_;
  OsqpEigen::Solver solver_;
  bool initialized_ = false;

  // 4 变量 QP ：P(4×4)  q(4)  A(8×4)  l/u(8)
  Eigen::SparseMatrix<double> P_;
  Eigen::SparseMatrix<double> A_;
  Eigen::Matrix<double, 4, 1> q_;
  Eigen::Matrix<double, 8, 1> l_;
  Eigen::Matrix<double, 8, 1> u_;
};

// ---------------------------------------------------------------------------
//  ROS 节点
// ---------------------------------------------------------------------------
class MotionRlShieldNode {
 public:
  explicit MotionRlShieldNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);

  void OnState(const std_msgs::Float32MultiArray::ConstPtr& msg);
  void Tick(const ros::TimerEvent&);

 private:
  bool LoadShieldYaml(const std::string& path);

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_state_;
  ros::Publisher  pub_cmd_;
  ros::Publisher  pub_shield_;
  ros::Timer      timer_;

  ShieldConfig                  shield_cfg_;
  std::unique_ptr<OnnxActor>    actor_;
  std::unique_ptr<MpcShield>    shield_;

  // 13 维系统状态最新值（双缓冲：sub 线程写、Tick 读）
  std::mutex                    state_mtx_;
  Eigen::Matrix<float, kStateDim, 1> latest_state_ = Eigen::Matrix<float, kStateDim, 1>::Zero();
  std::atomic<bool>             have_state_{false};

  // 上一拍真正施加的归一化动作（速率约束 / 平顺项使用）
  Eigen::Matrix<float, kActionDim, 1> u_prev_ = Eigen::Matrix<float, kActionDim, 1>::Zero();

  // 滚动告警窗口
  std::deque<uint8_t>           recent_hits_;
};

}  // namespace control
}  // namespace elane
