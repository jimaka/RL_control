// =============================================================================
//  RL-Actor + MPC-Shield  ::  Step 4 / 4
//  文件        : src/ship_control/src/motion_rl_shield.cpp
//  说明        :
//      ROS 节点入口 + ORT 推理 + OSQP 凸优化护盾 实现
//  作者        : RL-Shield team
// =============================================================================
#include "ship_control/motion_rl_shield.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

namespace elane {
namespace control {

namespace {
using Clock = std::chrono::steady_clock;
inline double ToMs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

// =============================================================================
//  OnnxActor : 零拷贝 ORT 推理
// =============================================================================
OnnxActor::OnnxActor(const std::string& model_path, bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "rl_shield_actor") {
  opts_.SetIntraOpNumThreads(1);
  opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (use_cuda) {
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    cuda_opts.arena_extend_strategy = 1;
    cuda_opts.gpu_mem_limit = SIZE_MAX;
    cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_opts.do_copy_in_default_stream = 1;
    try {
      opts_.AppendExecutionProvider_CUDA(cuda_opts);
      ROS_INFO("[ort] CUDAExecutionProvider 启用成功");
    } catch (const Ort::Exception& e) {
      ROS_WARN("[ort] CUDA 不可用，回落 CPU: %s", e.what());
    }
  }

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts_);

  // 一次性查询 IO 名字（不要每次推理都 query，会泄漏小块内存）
  Ort::AllocatorWithDefaultOptions alloc;
  in_name_str_  = std::string(session_->GetInputNameAllocated(0,  alloc).get());
  out_name_str_ = std::string(session_->GetOutputNameAllocated(0, alloc).get());
  in_names_[0]  = in_name_str_.c_str();
  out_names_[0] = out_name_str_.c_str();

  ROS_INFO("[ort] 模型已加载: %s  (in='%s' out='%s')",
           model_path.c_str(), in_names_[0], out_names_[0]);
}

double OnnxActor::Infer(const Eigen::Matrix<float, kStateDim, 1>& obs,
                        Eigen::Matrix<float, kActionDim, 1>* action) {
  obs_buf_ = obs;
  auto t0 = Clock::now();

  // 用预分配的 obs_buf_ 直接构造 Tensor（零拷贝）
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem_info_, obs_buf_.data(), static_cast<size_t>(kStateDim),
      in_shape_.data(), in_shape_.size());

  auto out_tensors = session_->Run(Ort::RunOptions{nullptr}, in_names_,
                                   &in_tensor, 1, out_names_, 1);
  // 出 tensor 直接 memcpy 到 act_buf_
  const float* raw = out_tensors.front().GetTensorData<float>();
  std::memcpy(act_buf_.data(), raw, sizeof(float) * kActionDim);
  *action = act_buf_;
  return ToMs(t0, Clock::now());
}

// =============================================================================
//  MpcShield : OSQP 4 变量 QP
// =============================================================================
MpcShield::MpcShield(const ShieldConfig& cfg) : cfg_(cfg) {
  BuildStaticMatrices();

  solver_.settings()->setVerbosity(false);
  solver_.settings()->setWarmStart(true);
  solver_.settings()->setMaxIteration(200);
  solver_.settings()->setAbsoluteTolerance(1e-5);
  solver_.settings()->setRelativeTolerance(1e-5);
  solver_.settings()->setPolish(true);

  solver_.data()->setNumberOfVariables(kActionDim);
  solver_.data()->setNumberOfConstraints(2 * kActionDim);  // 4 个盒 + 4 个速率盒
  if (!solver_.data()->setHessianMatrix(P_))    throw std::runtime_error("OSQP setHessianMatrix failed");
  if (!solver_.data()->setGradient(q_))         throw std::runtime_error("OSQP setGradient failed");
  if (!solver_.data()->setLinearConstraintsMatrix(A_)) throw std::runtime_error("OSQP setA failed");
  if (!solver_.data()->setLowerBound(l_))       throw std::runtime_error("OSQP setLB failed");
  if (!solver_.data()->setUpperBound(u_))       throw std::runtime_error("OSQP setUB failed");
  if (!solver_.initSolver())                    throw std::runtime_error("OSQP initSolver failed");
  initialized_ = true;
  ROS_INFO("[shield] OSQP 已初始化: nvar=%d ncon=%d", kActionDim, 2 * kActionDim);
}

void MpcShield::BuildStaticMatrices() {
  // P = 2 * (w_dev + w_smooth) * I  (来自 || u - u_rl ||² + w_smooth || u - u_prev ||²)
  P_.resize(kActionDim, kActionDim);
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(kActionDim);
  const double diag = 2.0 * (cfg_.w_dev + cfg_.w_smooth);
  for (int i = 0; i < kActionDim; ++i) trips.emplace_back(i, i, diag);
  P_.setFromTriplets(trips.begin(), trips.end());
  P_.makeCompressed();

  // A = [ I ;  I ]  →  上 4 行：盒约束， 下 4 行：速率盒约束
  A_.resize(2 * kActionDim, kActionDim);
  trips.clear();
  trips.reserve(2 * kActionDim);
  for (int i = 0; i < kActionDim; ++i) {
    trips.emplace_back(i,                i, 1.0);
    trips.emplace_back(i + kActionDim,   i, 1.0);
  }
  A_.setFromTriplets(trips.begin(), trips.end());
  A_.makeCompressed();

  q_.setZero();
  l_.setConstant(-1e30);
  u_.setConstant( 1e30);
}

void MpcShield::RefreshDynamic(const Eigen::Matrix<float, kActionDim, 1>& u_rl,
                               const Eigen::Matrix<float, kActionDim, 1>& u_prev) {
  // 梯度 q = -2 (w_dev * u_rl + w_smooth * u_prev)
  for (int i = 0; i < kActionDim; ++i) {
    q_(i) = -2.0 * (cfg_.w_dev    * static_cast<double>(u_rl(i))
                  + cfg_.w_smooth * static_cast<double>(u_prev(i)));
  }

  // 上 4 行：归一化盒约束
  for (int i = 0; i < kActionDim; ++i) {
    l_(i) = cfg_.abs_lo[i];
    u_(i) = cfg_.abs_hi[i];
  }

  // 下 4 行：速率盒约束（归一化空间）
  //   推力维 i=0,2  : rate_norm = thrust_rate_max_n_s / action_scale[i]
  //   舵   维 i=1,3 : rate_norm = rudder_rate_max_rad_s / action_scale[i]
  for (int i = 0; i < kActionDim; ++i) {
    const bool is_rudder = (i == 1 || i == 3);
    const double phys_rate = is_rudder ? cfg_.rudder_rate_max_rad_s
                                       : cfg_.thrust_rate_max_n_s;
    const double scale = cfg_.action_scale[i];
    const double rate_norm = (phys_rate * cfg_.control_dt) / std::max(scale, 1e-9);
    l_(i + kActionDim) = static_cast<double>(u_prev(i)) - rate_norm;
    u_(i + kActionDim) = static_cast<double>(u_prev(i)) + rate_norm;
  }
}

double MpcShield::Solve(const Eigen::Matrix<float, kActionDim, 1>& u_rl,
                        const Eigen::Matrix<float, kActionDim, 1>& u_prev,
                        Eigen::Matrix<float, kActionDim, 1>* u_safe,
                        uint8_t* bound_mask) {
  RefreshDynamic(u_rl, u_prev);

  auto t0 = Clock::now();
  if (!solver_.updateGradient(q_))       ROS_WARN_THROTTLE(2.0, "[shield] updateGradient failed");
  if (!solver_.updateBounds(l_, u_))     ROS_WARN_THROTTLE(2.0, "[shield] updateBounds failed");
  const auto status = solver_.solveProblem();
  const double qp_ms = ToMs(t0, Clock::now());

  if (status != OsqpEigen::ErrorExitFlag::NoError) {
    // 求解失败 → 安全回落：直接对 u_rl clip 到全局盒，并取与 u_prev 的速率
    Eigen::Vector4d fallback;
    for (int i = 0; i < kActionDim; ++i) {
      const double v = std::clamp<double>(u_rl(i), cfg_.abs_lo[i], cfg_.abs_hi[i]);
      fallback(i) = std::clamp<double>(v, l_(i + kActionDim), u_(i + kActionDim));
    }
    for (int i = 0; i < kActionDim; ++i) (*u_safe)(i) = static_cast<float>(fallback(i));
    *bound_mask = ComputeBoundMask(fallback, u_prev.cast<double>());
    ROS_WARN_THROTTLE(1.0, "[shield] OSQP 求解异常，已使用 clip 回退方案");
    return qp_ms;
  }

  const Eigen::VectorXd sol = solver_.getSolution();
  for (int i = 0; i < kActionDim; ++i) (*u_safe)(i) = static_cast<float>(sol(i));
  *bound_mask = ComputeBoundMask(sol, u_prev.cast<double>());
  return qp_ms;
}

uint8_t MpcShield::ComputeBoundMask(const Eigen::Vector4d& u_safe,
                                    const Eigen::Vector4d& u_prev) const {
  // bit0 推力盒, bit1 舵盒, bit2 转舵速率, bit3 推力速率
  constexpr double kEdgeEps = 1e-4;
  uint8_t mask = 0;
  for (int i = 0; i < kActionDim; ++i) {
    const bool is_rudder = (i == 1 || i == 3);
    if (std::abs(u_safe(i) - cfg_.abs_lo[i]) < kEdgeEps ||
        std::abs(u_safe(i) - cfg_.abs_hi[i]) < kEdgeEps) {
      mask |= is_rudder ? (1u << 1) : (1u << 0);
    }
    if (std::abs(u_safe(i) - l_(i + kActionDim)) < kEdgeEps ||
        std::abs(u_safe(i) - u_(i + kActionDim)) < kEdgeEps) {
      mask |= is_rudder ? (1u << 2) : (1u << 3);
    }
  }
  return mask;
}

// =============================================================================
//  MotionRlShieldNode : ROS 入口
// =============================================================================
MotionRlShieldNode::MotionRlShieldNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {

  std::string policy_path, config_file, topic_state, topic_cmd;
  bool use_cuda = true;
  double rate_hz = 10.0;
  pnh_.param<std::string>("policy_path", policy_path, "");
  pnh_.param<std::string>("config_file", config_file, "");
  pnh_.param<bool>       ("use_cuda",    use_cuda,    true);
  pnh_.param<double>     ("rate_hz",     rate_hz,     10.0);
  pnh_.param<std::string>("topic_state", topic_state, "/ship/state13");
  pnh_.param<std::string>("topic_cmd",   topic_cmd,   "/control/cmd");

  if (policy_path.empty()) {
    ROS_FATAL("[node] policy_path 为空，无法启动");
    ros::shutdown();
    return;
  }
  if (!LoadShieldYaml(config_file)) {
    ROS_WARN("[node] 未加载 shield_bounds.yaml，使用代码内默认值");
  }

  actor_  = std::make_unique<OnnxActor>(policy_path, use_cuda);
  shield_ = std::make_unique<MpcShield>(shield_cfg_);

  sub_state_  = nh_.subscribe<std_msgs::Float32MultiArray>(
      topic_state, 1, &MotionRlShieldNode::OnState, this);
  pub_cmd_    = nh_.advertise<elane_msgs::ControlCmd>(topic_cmd, 1);
  pub_shield_ = nh_.advertise<ship_control::ShieldIntervention>(
      "/control/shield_intervention", 10);

  timer_ = nh_.createTimer(ros::Duration(1.0 / rate_hz),
                           &MotionRlShieldNode::Tick, this);
  ROS_INFO("[node] motion_rl_shield_node 已启动: rate=%.1fHz state=%s cmd=%s",
           rate_hz, topic_state.c_str(), topic_cmd.c_str());
}

bool MotionRlShieldNode::LoadShieldYaml(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream fin(path);
  if (!fin.good()) {
    ROS_WARN("[node] 找不到 YAML: %s", path.c_str());
    return false;
  }
  try {
    YAML::Node root = YAML::LoadFile(path);
    if (root["control_dt"]) shield_cfg_.control_dt = root["control_dt"].as<double>();

    if (root["action_scale"]) {
      auto arr = root["action_scale"];
      for (int i = 0; i < kActionDim && i < static_cast<int>(arr.size()); ++i)
        shield_cfg_.action_scale[i] = arr[i].as<double>();
    }
    if (root["abs_bounds"]) {
      auto arr = root["abs_bounds"];
      for (int i = 0; i < kActionDim && i < static_cast<int>(arr.size()); ++i) {
        shield_cfg_.abs_lo[i] = arr[i][0].as<double>();
        shield_cfg_.abs_hi[i] = arr[i][1].as<double>();
      }
    }
    if (root["rate_limits"]) {
      auto rl = root["rate_limits"];
      if (rl["rudder_rate_max_rad_s"])
        shield_cfg_.rudder_rate_max_rad_s = rl["rudder_rate_max_rad_s"].as<double>();
      if (rl["thrust_rate_max_n_s"])
        shield_cfg_.thrust_rate_max_n_s = rl["thrust_rate_max_n_s"].as<double>();
    }
    if (root["qp_weights"]) {
      auto qw = root["qp_weights"];
      if (qw["w_dev"])    shield_cfg_.w_dev    = qw["w_dev"].as<double>();
      if (qw["w_smooth"]) shield_cfg_.w_smooth = qw["w_smooth"].as<double>();
    }
    if (root["shield"]) {
      auto sh = root["shield"];
      if (sh["trigger_eps"]) shield_cfg_.trigger_eps = sh["trigger_eps"].as<double>();
      if (sh["warn_rate"])   shield_cfg_.warn_rate   = sh["warn_rate"].as<double>();
      if (sh["warn_window"]) shield_cfg_.warn_window = sh["warn_window"].as<int>();
    }
    ROS_INFO("[node] shield_bounds.yaml 已加载: dt=%.3f w_dev=%.2f w_smooth=%.2f",
             shield_cfg_.control_dt, shield_cfg_.w_dev, shield_cfg_.w_smooth);
    return true;
  } catch (const std::exception& e) {
    ROS_ERROR("[node] YAML 解析失败: %s", e.what());
    return false;
  }
}

void MotionRlShieldNode::OnState(const std_msgs::Float32MultiArray::ConstPtr& msg) {
  if (msg->data.size() < kStateDim) {
    ROS_WARN_THROTTLE(2.0, "[node] state 维度不足: %zu < %d",
                      msg->data.size(), kStateDim);
    return;
  }
  std::lock_guard<std::mutex> lk(state_mtx_);
  for (int i = 0; i < kStateDim; ++i) latest_state_(i) = msg->data[i];
  have_state_.store(true, std::memory_order_release);
}

void MotionRlShieldNode::Tick(const ros::TimerEvent&) {
  if (!have_state_.load(std::memory_order_acquire)) return;

  Eigen::Matrix<float, kStateDim, 1> obs;
  {
    std::lock_guard<std::mutex> lk(state_mtx_);
    obs = latest_state_;
  }

  // ---- 1) ORT 推理 -------------------------------------------------------
  Eigen::Matrix<float, kActionDim, 1> u_rl;
  const double infer_ms = actor_->Infer(obs, &u_rl);

  // ---- 2) OSQP 护盾 ------------------------------------------------------
  Eigen::Matrix<float, kActionDim, 1> u_safe;
  uint8_t bound_mask = 0;
  const double qp_ms = shield_->Solve(u_rl, u_prev_, &u_safe, &bound_mask);

  // ---- 3) 干预残差 -------------------------------------------------------
  Eigen::Matrix<float, kActionDim, 1> diff = u_safe - u_rl;
  const float e_shield = diff.squaredNorm();
  const bool active = e_shield > static_cast<float>(shield_cfg_.trigger_eps);

  // ---- 4) 物理量纲反归一化 ------------------------------------------------
  std::array<float, kActionDim> u_safe_phys{};
  for (int i = 0; i < kActionDim; ++i) {
    u_safe_phys[i] = static_cast<float>(u_safe(i) * shield_cfg_.action_scale[i]);
  }

  // ---- 5) 发布物理控制指令（与 motion.cpp ControlCmd 字段对齐） -------------
  elane_msgs::ControlCmd cmd;
  cmd.port_thruster_throttle      = u_safe_phys[0];
  cmd.port_thruster_angle         = u_safe_phys[1] * 180.0f / static_cast<float>(M_PI);
  cmd.starboard_thruster_throttle = u_safe_phys[2];
  cmd.starboard_thruster_angle    = u_safe_phys[3] * 180.0f / static_cast<float>(M_PI);
  pub_cmd_.publish(cmd);

  // ---- 6) 发布护盾干预消息 ------------------------------------------------
  ship_control::ShieldIntervention si;
  si.header.stamp = ros::Time::now();
  si.e_shield     = e_shield;
  si.active       = active;
  si.bound_mask   = bound_mask;
  for (int i = 0; i < kActionDim; ++i) {
    si.u_rl[i]        = u_rl(i);
    si.u_safe[i]      = u_safe(i);
    si.u_safe_phys[i] = u_safe_phys[i];
  }
  si.infer_ms = static_cast<float>(infer_ms);
  si.qp_ms    = static_cast<float>(qp_ms);
  pub_shield_.publish(si);

  // ---- 7) 滚动告警 -------------------------------------------------------
  recent_hits_.push_back(active ? 1u : 0u);
  if (static_cast<int>(recent_hits_.size()) > shield_cfg_.warn_window) {
    recent_hits_.pop_front();
  }
  if (static_cast<int>(recent_hits_.size()) >= shield_cfg_.warn_window) {
    const double hit_rate =
        static_cast<double>(std::accumulate(recent_hits_.begin(),
                                            recent_hits_.end(), 0)) /
        static_cast<double>(recent_hits_.size());
    if (hit_rate > shield_cfg_.warn_rate) {
      ROS_WARN_THROTTLE(2.0,
          "[shield] 高干预率 %.1f%% (window=%d)  E_shield=%.4f  mask=0x%02X "
          "→ RL 策略可能失稳，建议下线",
          hit_rate * 100.0, shield_cfg_.warn_window, e_shield, bound_mask);
    }
  }

  u_prev_ = u_safe;  // 真实施加给船的就是 u_safe
}

}  // namespace control
}  // namespace elane

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_rl_shield_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  try {
    elane::control::MotionRlShieldNode node(nh, pnh);
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL("[main] 节点崩溃: %s", e.what());
    return 1;
  }
  return 0;
}
