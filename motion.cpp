#include "motion.h"

#include "../mpc/glo_data.hpp"
#include "../mpc/mpc_AD.h"
#include "force.hpp"
#include "para.h"

namespace elane {

namespace control {

ControlNode::ControlNode(ros::NodeHandle &nh)
    : subs_(static_cast<int>(SubscriberId::SUBSCRIBER_ID_LENGTH)),
      pubs_(static_cast<int>(PublisherId::PUBLISHER_ID_LENGTH)) {
  module_result_publisher_ =
      std::make_shared<botix::common::ModulesResultsPublisher>(
          nh, botix::common::ModulesFrameID::CONTROL_MODULE);

  subs_[static_cast<int>(SubscriberId::TRAJECTORY)] =
      nh.subscribe<elane_msgs::Trajectory>("/planning/trajectory", 1,
                                           &ControlNode::SubTrajectory, this);
  subs_[static_cast<int>(SubscriberId::FUSIONPOSESTAMPED)] =
      nh.subscribe<elane_msgs::FusionPoseStamped>(
          "/localization/fusion_pose", 1, &ControlNode::SubFusionPoseStamped,
          this);

  subs_[static_cast<int>(SubscriberId::UNDER_STATE)] =
      nh.subscribe<elane_msgs::ControlCmd>("/state/controller_state", 1,
                                           &ControlNode::SubUnderState, this);

  subs_[static_cast<int>(SubscriberId::THRUSTER_FEEDBACK)] =
      nh.subscribe<elane_msgs::ThrusterFeedback>(
          "/system/chassis_feedback", 1, &ControlNode::SubThrusterFeedback,
          this);
  subs_[static_cast<int>(SubscriberId::ONLIE_PARA)] =
      nh.subscribe<ship_hydro_model_pkg::MiddleHydroParams>(
          "/draught_observation_results_topic", 1, &ControlNode::SubOnlinePara,
          this);

  pubs_[static_cast<int>(PublisherId::CONTROL_CMD)] =
      nh.advertise<elane_msgs::ControlCmd>("/control/control_cmd", 1);

  pubs_[static_cast<int>(PublisherId::DEBUG_MSG)] =
      nh.advertise<ship_control::control_mpc_debug>("/control/debug_msg", 1);

  timer_ = nh.createTimer(ros::Duration(0.5), &ControlNode::Run, this);

  Init();
}

void ControlNode::Init() {
  LOGGER_INIT("control/ship_control.log");
  YamlInit("/opt/elane/ros/share/ship_control/config/para.yaml");
  ship_id_ = config_["ship_id_"].as<int32_t>();
  throttle_back_gain_ = config_["throttle_back_gain"].as<double>();
  kappa_gain_ = config_["kappa_gain_"].as<double>();
  ship_deadweight_ = config_["ship_deadweight_"].as<int>();
  elane::control::Config::instance().set_ship_deadweight(ship_deadweight_);
  printf("ship_id_ %d throttle_back_gain_ %lf  kappa_gain_ %lf\n", ship_id_,
         throttle_back_gain_, kappa_gain_);
  printf("ship_deadweight_ %d \n", ship_deadweight_);
  xicheng_azimuthing_MpcTaRudderSlove_.init();
}

void ControlNode::YamlInit(std::string file_path) {
  try {
    config_ = YAML::LoadFile(file_path);
  } catch (YAML::BadFile &e) {
    std::cout << std::string(e.what()) << std::endl;
  }

  std::string log_path = config_["log_path"].as<std::string>();

  printf("path == %s \n", log_path.c_str());
  const std::string ship_info_file_name =
      "/opt/elane/config/ship_info/ship_info.yaml";
  YAML::Node ship_info_config = YAML::LoadFile(ship_info_file_name);
  const std::string ship_name = ship_info_config["ship_name"].as<std::string>();
  // 2. load dynamic parameters by ship name
  std::string dynamic_param_file_name =
      "/opt/elane/config/ship_info/botix_params/dynamic_param_files/" +
      ship_name + ".yaml";
  YAML::Node dynamic_params_config = YAML::LoadFile(dynamic_param_file_name);
  double length = dynamic_params_config["ship_length"].as<double>();
  double width = dynamic_params_config["ship_width"].as<double>();
  double front_edge_to_center =
      dynamic_params_config["sensors_extrinsic_param"]
                           ["rotation_center_ship_front_edge"]
                               .as<double>();
  double fusion_pose_to_ship_front_edge =
      dynamic_params_config["sensors_extrinsic_param"]
                           ["fusion_pose_to_ship_front_edge"]
                               .as<double>();
  double fusion_pose_ship_left_edge =
      dynamic_params_config["sensors_extrinsic_param"]
                           ["fusion_pose_ship_left_edge"]
                               .as<double>();
  double rotation_center_ship_left_edge =
      dynamic_params_config["sensors_extrinsic_param"]
                           ["rotation_center_ship_left_edge"]
                               .as<double>();
  rotation_center_distance_ =
      front_edge_to_center - fusion_pose_to_ship_front_edge;
  rotation_center_distance_lat_ =
      fusion_pose_ship_left_edge - rotation_center_ship_left_edge;

  printf("rotation_center_distance_ %lf \n", rotation_center_distance_);
  printf("rotation_center_distance_lat_ %lf \n", rotation_center_distance_lat_);
}

void ControlNode::SubTrajectory(const elane_msgs::Trajectory::ConstPtr &msg) {
  Trajectory_ = std::move(msg);
  parking_flag = msg->is_parking;
  if (false == remote_flag_) {
    run_flag_ = true;
  }
  count_[static_cast<int>(CountId::PLAN_COUNT)] = 0;
  los_pid_.SetTrajectory(Trajectory_);
  double final_point_index = Trajectory_->trajectory_points.size() - 1;
  elane::control::Config::instance().set_final_point_speed(
      Trajectory_->trajectory_points[final_point_index].v);
  elane::control::Config::instance().get_final_point_speed(final_speed_);
  printf("final_point_speed %lf \n", final_speed_);
}

void ControlNode::SubFusionPoseStamped(
    const elane_msgs::FusionPoseStamped::ConstPtr &msg) {
  FusionPoseStamped_ = std::move(msg);
  los_pid_.SetFusionPoseStamped(FusionPoseStamped_);
  rotation_center_x_ =
      FusionPoseStamped_->position.x -
      rotation_center_distance_lat_ * sin(FusionPoseStamped_->rpy.yaw) -
      rotation_center_distance_ * cos(FusionPoseStamped_->rpy.yaw);
  rotation_center_y_ =
      FusionPoseStamped_->position.y -
      rotation_center_distance_lat_ * cos(FusionPoseStamped_->rpy.yaw) -
      rotation_center_distance_ * sin(FusionPoseStamped_->rpy.yaw);
  // printf("rotation_center_x_ %lf rotation_center_y_ %lf \n",
  // rotation_center_x_,
  //        rotation_center_y_);
  count_[static_cast<int>(CountId::NAV_COUNT)] = 0;
  if (0 != nav_t_) {
    if ((msg->header.stamp.toSec() - nav_t_) > 0.1) {
      double dt = (msg->header.stamp.toSec() - nav_t_);
      double nav_yaw_acc_ = (msg->angular_velocity.z - nav_yaw_rate_) / (dt);
      nav_t_ = msg->header.stamp.toSec();
      x_acc_ = (msg->velocity.x - x_v_) / dt;
      y_acc_ = (msg->velocity.y - y_v_) / dt;
      yaw_rate_ = (msg->rpy.yaw - yaw_) / dt;
      yaw_rate_acc_ = (yaw_rate_ - yaw_rate_old_) / dt;
      x_v_ = msg->velocity.x;
      y_v_ = msg->velocity.y;
      yaw_ = msg->rpy.yaw;
      yaw_rate_old_ = yaw_rate_;
      x_acc_ = msg->linear_acceleration.x;
      double x_state_u_dot_vr =
          x_acc_ - msg->angular_velocity.z * msg->velocity.y;
      double x_state_u_dot = x_acc_;
      double x_state_u = msg->velocity.x;
      double x_state_uu = msg->velocity.x * abs(msg->velocity.x);
      y_acc_ = msg->linear_acceleration.y;
      double y_state_v_dot_ur =
          y_acc_ + msg->angular_velocity.z * msg->velocity.x;
      double y_state_v_dot = y_acc_;
      double y_state_v = msg->velocity.y;
      double y_state_vv = msg->velocity.y * abs(msg->velocity.y);
      double z_state_r_dot = nav_yaw_acc_;
      double z_state_r = msg->angular_velocity.z;
      double z_state_rr =
          msg->angular_velocity.z * abs(msg->angular_velocity.z);
      // rls_.load_state(x_state_u_dot_vr, x_state_u_dot, x_state_u, x_state_uu,
      //                 y_state_v_dot_ur, y_state_v_dot, y_state_v, y_state_vv,
      //                 z_state_r_dot, z_state_r, z_state_rr);
      // rls_.update_with_forget_factor(0.97);
      // nav_yaw_rate_ = msg->angular_velocity.z;
      // rls_.update();
      // printf("dt %lf nav_yaw_acc_ %lf  old %lf new %lf \n", dt, nav_yaw_acc_,
      //        nav_yaw_rate_, msg->angular_velocity.z);
    }
  }
  if (0 == nav_t_) {
    nav_t_ = msg->header.stamp.toSec();
    x_v_ = msg->velocity.x;
    y_v_ = msg->velocity.y;
    yaw_ = msg->rpy.yaw;
    nav_yaw_rate_ = msg->angular_velocity.z;
  }
}

void ControlNode::SubUnderState(const elane_msgs::ControlCmd::ConstPtr &msg) {
  under_state_ = std::move(msg);

  // printf("port_thruster_angle %lf starboard_thruster_angle %lf \n",
  //        static_cast<double>(msg->port_thruster_angle) * M_PI / 180,
  //        static_cast<double>(msg->starboard_thruster_angle) * M_PI / 180);
  double port_angle =
      static_cast<double>(msg->port_thruster_angle) * M_PI / 180;
  double starboard_angle =
      static_cast<double>(msg->starboard_thruster_angle) * M_PI / 180;

  double x_force =
      under_state_->port_thruster_throttle * cos(port_angle) * force_tnn_ +
      under_state_->starboard_thruster_throttle * cos(starboard_angle) *
          force_tnn_;
  double y_force =
      under_state_->port_thruster_throttle * sin(port_angle) * force_tnn_ +
      under_state_->starboard_thruster_throttle * sin(starboard_angle) *
          force_tnn_;
  double z_force =
      force_tnn_ * 4.5 *
          (under_state_->starboard_thruster_throttle * cos(starboard_angle) -
           under_state_->port_thruster_throttle * cos(port_angle)) -
      force_tnn_ * 44 *
          (under_state_->starboard_thruster_throttle * sin(starboard_angle) +
           under_state_->port_thruster_throttle * sin(port_angle));
  // printf("x_force %lf y_force %lf z_force %lf \n", x_force, y_force,
  // z_force);
  rls_.load_force(x_force, y_force, z_force);
}

void ControlNode::SubThrusterFeedback(
    const elane_msgs::ThrusterFeedback::ConstPtr &msg) {
  printf("SubThrusterFeedback \n");
  thruster_feedback_ = std::move(msg);
  // if (thruster_feedback_) {
  //   double t1_force = 0;
  //   double t2_force = 0;
  //   ThrottleToForce(
  //       static_cast<double>(thruster_feedback_->port_thruster_throttle),
  //       t1_force);
  //   ThrottleToForce(
  //       static_cast<double>(thruster_feedback_->starboard_thruster_throttle),
  //       t2_force);
  //   printf("throttle lf %lf rf %lf \n",
  //          thruster_feedback_->port_thruster_throttle,
  //          thruster_feedback_->starboard_thruster_throttle);
  //   xicheng_azimuthing_MpcTaRudderSlove_.set_xp1_force_init(t1_force);
  //   xicheng_azimuthing_MpcTaRudderSlove_.set_xp2_force_init(t2_force);
  //   printf("t1_force %lf t2_force %lf \n", t1_force, t2_force);
  //   xicheng_azimuthing_MpcTaRudderSlove_.set_xp1_angle_init(
  //       thruster_feedback_->port_thruster_angle * M_PI / 180);
  //   xicheng_azimuthing_MpcTaRudderSlove_.set_xp2_angle_init(
  //       thruster_feedback_->starboard_thruster_angle * M_PI / 180);
  //   elane::control::ConfigThrustSend::instance().set_port_thruster_force(
  //       t1_force);
  //   elane::control::ConfigThrustSend::instance().set_starboard_thruster_force(
  //       t2_force);
  //   elane::control::ConfigThrustSend::instance().set_port_thruster_angle(
  //       thruster_feedback_->port_thruster_angle * M_PI / 180);
  //   elane::control::ConfigThrustSend::instance().set_starboard_thruster_angle(
  //       thruster_feedback_->starboard_thruster_angle * M_PI / 180);

  //   printf("mpc input lf %lf la %lf rf %lf ra %lf \n", t1_force,
  //          thruster_feedback_->port_thruster_angle * M_PI / 180, t2_force,
  //          thruster_feedback_->starboard_thruster_angle * M_PI / 180);
  // }
}

void ControlNode::SubJoy(const sensor_msgs::Joy::ConstPtr &msg) {
  if (1 == msg->buttons[0]) {
    // under_data_.control_source = auto_control_mode;
    remote_flag_ = false;
  }

  if (1 == msg->buttons[1]) {
    // under_data_.control_source = remote_mode;
    remote_flag_ = true;
    run_flag_ = false;
  }
}

void ControlNode::SubOnlinePara(
    const ship_hydro_model_pkg::MiddleHydroParams::ConstPtr &msg) {
  printf("SubOnlinePara \n");
  online_para_ = std::move(msg);
  elane::control::Config::instance().set_online_para_flag(true);
  elane::control::Config::instance().set_u_u(msg->H_u_n_u);
  elane::control::Config::instance().set_u_uu(msg->H_u_n_uu);
  elane::control::Config::instance().set_u_uu_cf(msg->H_u_n_uu_Cf);
  elane::control::Config::instance().set_u_rr(msg->H_u_n_rr);
  elane::control::Config::instance().set_u_f_X(msg->H_u_n_tau_X);

  printf(
      "H_u_n_u %lf H_u_n_uu %lf H_u_n_uu_Cf %lf H_u_n_rr %lf H_u_n_tau_X %lf "
      "\n",
      msg->H_u_n_u, msg->H_u_n_uu, msg->H_u_n_uu_Cf, msg->H_u_n_rr,
      msg->H_u_n_tau_X);

  elane::control::Config::instance().set_v_ur(msg->H_v_n_ur);
  elane::control::Config::instance().set_v_uv(msg->H_v_n_uv);
  elane::control::Config::instance().set_v_vV(msg->H_v_n_vV);
  elane::control::Config::instance().set_v_vv(msg->H_v_n_vv);
  elane::control::Config::instance().set_v_vr(msg->H_v_n_vr);
  elane::control::Config::instance().set_v_r_V(msg->H_v_n_rV);
  elane::control::Config::instance().set_v_rv(msg->H_v_n_rv);
  elane::control::Config::instance().set_v_rr(msg->H_v_n_rr);
  elane::control::Config::instance().set_v_f_Y(msg->H_v_n_tau_Y);
  elane::control::Config::instance().set_v_f_N(msg->H_v_n_tau_N);

  printf(
      "H_v_n_ur %lf H_v_n_uv %lf H_v_n_vV %lf H_v_n_vv %lf H_v_n_vr %lf "
      "H_v_n_rV %lf H_v_n_rv %lf H_v_n_rr %lf H_v_n_tau_Y %lf H_v_n_tau_N "
      "%lf \n",
      msg->H_v_n_ur, msg->H_v_n_uv, msg->H_v_n_vV, msg->H_v_n_vv, msg->H_v_n_vr,
      msg->H_v_n_rV, msg->H_v_n_rv, msg->H_v_n_rr, msg->H_v_n_tau_Y,
      msg->H_v_n_tau_N);

  elane::control::Config::instance().set_r_ur(msg->H_r_n_ur);
  elane::control::Config::instance().set_r_uv(msg->H_r_n_uv);
  elane::control::Config::instance().set_r_vV(msg->H_r_n_vV);
  elane::control::Config::instance().set_r_vv(msg->H_r_n_vv);
  elane::control::Config::instance().set_r_vr(msg->H_r_n_vr);
  elane::control::Config::instance().set_r_rV(msg->H_r_n_rV);
  elane::control::Config::instance().set_r_rv(msg->H_r_n_rv);
  elane::control::Config::instance().set_r_rr(msg->H_r_n_rr);
  elane::control::Config::instance().set_r_f_Y(msg->H_r_n_tau_Y);
  elane::control::Config::instance().set_r_f_N(msg->H_r_n_tau_N);
  printf(
      "H_r_n_ur %lf H_r_n_uv %lf H_r_n_vV %lf H_r_n_vv %lf H_r_n_vr %lf "
      "H_r_n_rV %lf H_r_n_rv %lf H_r_n_rr %lf H_r_n_tau_Y %lf H_r_n_tau_N "
      "%lf \n",
      msg->H_r_n_ur, msg->H_r_n_uv, msg->H_r_n_vV, msg->H_r_n_vv, msg->H_r_n_vr,
      msg->H_r_n_rV, msg->H_r_n_rv, msg->H_r_n_rr, msg->H_r_n_tau_Y,
      msg->H_r_n_tau_N);
}

void ControlNode::ShipBaseChange(const double target_x, const double target_y,
                                 const double target_yaw, double &ship_base_x,
                                 double &ship_base_y, double &ship_base_yaw) {
  double ship_base_target_yaw = 0;
  Eigen::Vector3d ship_base_target_point;
  Eigen::Quaterniond ship_base_Q;
  Eigen::Vector3d enu_point = enuPoint(target_x, target_y, 0);
  Eigen::Quaterniond enu_point_att = enuPointAttitude(0, 0, target_yaw);
  if (FusionPoseStamped_) {
    Eigen::Vector3d enu_ship =
        enuPoint(rotation_center_x_, rotation_center_y_, 0);
    Eigen::Quaterniond enu_ship_att;
    enu_ship_att = enuPointAttitude(0, 0, FusionPoseStamped_->rpy.yaw);
    transformENUToCar(enu_point, enu_point_att, enu_ship, enu_ship_att,
                      ship_base_target_point, ship_base_Q,
                      ship_base_target_yaw);
    ship_base_x = ship_base_target_point.x();
    ship_base_y = ship_base_target_point.y();
    ship_base_yaw = ship_base_target_yaw;
  }
}

void ControlNode::MpcTaRunShunYi() {
  Eigen::VectorXd ini_state(6);
  target_data target_data_0;
  if (FusionPoseStamped_) {
    ini_state << 0, 0, 0, FusionPoseStamped_->velocity.x,
        FusionPoseStamped_->velocity.y, FusionPoseStamped_->angular_velocity.z;
    MpcTaSlove_.Solve(ini_state, target_data_0);
  }

  PublishControlCommand();
  PublishDebugInfo();
}

void ControlNode::MpcTaRunXiCheng() {
  Eigen::VectorXd ini_state(6);
  target_data target_data_0;
  if (thruster_feedback_) {
    double t1_force = 0;
    double t2_force = 0;
    ThrottleToForce(
        static_cast<double>(thruster_feedback_->port_thruster_throttle),
        t1_force);
    ThrottleToForce(
        static_cast<double>(thruster_feedback_->starboard_thruster_throttle),
        t2_force);
    xicheng_azimuthing_MpcTaRudderSlove_.set_xp1_force_init(t1_force);
    xicheng_azimuthing_MpcTaRudderSlove_.set_xp2_force_init(t2_force);
    printf("t1_force %lf t2_force %lf \n", t1_force, t2_force);
    // xicheng_azimuthing_MpcTaRudderSlove_.set_xp1_angle_init(
    //     thruster_feedback_->port_thruster_angle * M_PI / 180);
    // xicheng_azimuthing_MpcTaRudderSlove_.set_xp2_angle_init(
    //     thruster_feedback_->starboard_thruster_angle * M_PI / 180);
    elane::control::ConfigThrustSend::instance().set_port_thruster_force(
        t1_force);
    elane::control::ConfigThrustSend::instance().set_starboard_thruster_force(
        t2_force);
    elane::control::ConfigThrustSend::instance().set_port_thruster_angle(
        thruster_feedback_->port_thruster_angle * M_PI / 180);
    elane::control::ConfigThrustSend::instance().set_starboard_thruster_angle(
        thruster_feedback_->starboard_thruster_angle * M_PI / 180);

    printf("mpc input lf %lf la %lf rf %lf ra %lf \n", t1_force,
           thruster_feedback_->port_thruster_angle * M_PI / 180, t2_force,
           thruster_feedback_->starboard_thruster_angle * M_PI / 180);
  }

  if (FusionPoseStamped_ && Trajectory_) {
    PointChange();
    printf("PointChange done \n");
    for (size_t i = 0; i < mpc_steps; i++) {
      printf("target x %lf y %lf psi %lf u %lf  v %lf \n",
             mpc_states_gl.targets[i].x, mpc_states_gl.targets[i].y,
             mpc_states_gl.targets[i].psi, mpc_states_gl.targets[i].u,
             mpc_states_gl.targets[i].v);
    }
  }

  if (FusionPoseStamped_) {
    // ini_state << 0, 0, 0, FusionPoseStamped_->velocity.x,
    //     FusionPoseStamped_->velocity.y,
    //     FusionPoseStamped_->angular_velocity.z;
    ini_state << 0, 0, 0, FusionPoseStamped_->velocity.x,
        FusionPoseStamped_->velocity.y, yaw_rate_;
    // ini_state << 0, 0, 0, 0, 0, 0;
    // std::cout << "ini_state" << ini_state << std::endl;
    xicheng_azimuthing_MpcTaRudderSlove_.Solve(ini_state, target_data_0);
  }

  PublishControlCommand();
  PublishDebugInfo();
}
void ControlNode::PublishControlCommand() {
  auto msg = elane_msgs::ControlCmd();
  msg.header.stamp = ros::Time::now();
  ThrustReverse();
  if (abs(thrust_command_send.t1_angle) > 90) {
    thrust_command_send.t1_force =
        thrust_command_send.t1_force * throttle_back_gain_;
    thrust_command_send.t1_angle += 2;
    if (abs(thrust_command_send.t1_angle) > 180) {
      thrust_command_send.t1_angle -= 360;
    }
  } else {
    thrust_command_send.t1_angle += 2;
  }

  if (abs(thrust_command_send.t2_angle) > 90) {
    thrust_command_send.t2_force =
        thrust_command_send.t2_force * throttle_back_gain_;
    thrust_command_send.t2_angle += 2;
    if (abs(thrust_command_send.t2_angle) > 180) {
      thrust_command_send.t2_angle -= 360;
    }
  } else {
    thrust_command_send.t2_angle += 2;
  }
  msg.port_thruster_throttle = thrust_command_send.t1_force;
  msg.starboard_thruster_throttle = thrust_command_send.t2_force;
  msg.port_thruster_angle = thrust_command_send.t1_angle;
  msg.starboard_thruster_angle = thrust_command_send.t2_angle;
  printf("throttle1 %lf throttle2 %lf angle1 %lf angle2 %lf \n",
         thrust_command_send.t1_force, thrust_command_send.t2_force,
         thrust_command_send.t1_angle, thrust_command_send.t2_angle);
  pubs_[static_cast<int>(PublisherId::CONTROL_CMD)].publish(msg);
}

void ControlNode::PublishDebugInfo() {
  auto debug_out_msg = ship_control::control_mpc_debug();
  debug_out_msg.targets.clear();
  debug_out_msg.actions.clear();
  debug_out_msg.targets.reserve(mpc_steps);
  debug_out_msg.actions.reserve(mpc_steps);
  debug_out_msg = std::move(mpc_states_gl);
  pubs_[static_cast<int>(PublisherId::DEBUG_MSG)].publish(debug_out_msg);
}
void ControlNode::Run(const ros::TimerEvent &time_e) {
  // printf("control run \n");
  DisconnectJudge();

  if (true == run_flag_) {
    switch (ship_id_) {
      case static_cast<int>(ThrustModeId::DOUBLE_THRUST_SHUN_YI):
        MpcTaRunShunYi();
        // printf("MpcTaRunShunYi(); \n");
        break;
      case static_cast<int>(ThrustModeId::DOUBLE_THRUST_XI_CHENG):
        MpcTaRunXiCheng();
        // printf("MpcTaRunXiCheng(); \n");
        break;

      default:
        break;
    }
  }
}

void ControlNode::DisconnectJudge() {
  for (int i = 0; i < static_cast<int>(CountId::COUNT_ID_LENGTH); i++) {
    if (count_[i] < 255) {
      count_[i]++;
    }
  }

  if (count_[static_cast<int>(CountId::NAV_COUNT)] > 50) {
    // AERROR("nav disconnect");
    // printf("nav disconnect \n");
    Stop();
  }

  if (count_[static_cast<int>(CountId::PLAN_COUNT)] > 40) {
    // AERROR("planning_disconnect");
    // printf("planning_disconnect \n");
    if (run_flag_) {
      run_flag_ = false;
      Stop();
    }
  }
}

void ControlNode::Stop() {
  auto msg = elane_msgs::ControlCmd();
  msg.header.stamp = ros::Time::now();
  msg.port_thruster_throttle = 0;
  msg.starboard_thruster_throttle = 0;
  msg.port_thruster_angle = 0;
  msg.starboard_thruster_angle = 0;
  pubs_[static_cast<int>(PublisherId::CONTROL_CMD)].publish(msg);
  // printf("stop the ship \n");
}

void ControlNode::ThrustReverse() {
  for (size_t i = 0; i < 19; i++) {
    if (thrust_command_send.t1_force < force_spring[i + 1] &&
        thrust_command_send.t1_force >= force_spring[i]) {
      thrust_command_send.t1_force =
          i * 5 + ((thrust_command_send.t1_force - force_spring[i]) * 5 /
                   (force_spring[i + 1] - force_spring[i]));
      break;
    }
  }
  for (size_t i = 0; i < 19; i++) {
    if (thrust_command_send.t2_force < force_spring[i + 1] &&
        thrust_command_send.t2_force >= force_spring[i]) {
      thrust_command_send.t2_force =
          i * 5 + ((thrust_command_send.t2_force - force_spring[i]) * 5 /
                   (force_spring[i + 1] - force_spring[i]));
      break;
    }
  }
}

void ControlNode::PointChange() {
  auto targets_data_store = ship_control::mpc_state();
  planning_points_new.clear();
  planning_points_orin.clear();
  mpc_states_gl.targets.clear();
  mpc_states_gl.targets.reserve(mpc_steps);

  for (size_t i = 0; i < Trajectory_->trajectory_points.size(); i++) {
    planning_point_temp.SetX(Trajectory_->trajectory_points[i].x);
    planning_point_temp.SetY(Trajectory_->trajectory_points[i].y);
    planning_point_temp.SetPsi(Trajectory_->trajectory_points[i].theta);
    planning_point_temp.SetV(Trajectory_->trajectory_points[i].v *
                             Trajectory_->trajectory_points[i].gear);
    planning_point_temp.SetEpsi(Trajectory_->trajectory_points[i].dtheta);
    planning_point_temp.SetT(Trajectory_->trajectory_points[i].t);
    planning_point_temp.SetKappa(Trajectory_->trajectory_points[i].kappa);
    planning_points_orin.emplace_back(planning_point_temp);
  }
  get_point.StorePoints(planning_points_orin);

  double kappa_mean = 0;
  double acc_mean = 0;

  for (size_t i = 0; i < mpc_steps; i++) {
    get_point.GetPlanningPointWithTime(i * mpc_during + 0.5,
                                       &planning_point_temp);
    double x, y, yaw;
    ShipBaseChange(planning_point_temp.X(), planning_point_temp.Y(),
                   planning_point_temp.Psi(), x, y, yaw);
    target_datas[i].x = x;
    target_datas[i].y = y;
    target_datas[i].theta = yaw;
    target_datas[i].v = planning_point_temp.V();
    // target_datas[i].v = planning_point_temp.V()*planning_point_temp.V() -

    target_datas[i].v = planning_point_temp.V();
    if (final_speed_ != 0) {
      target_datas[i].v_y = target_datas[i].y / ((i + 1) * mpc_during);
      target_datas[i].theta += atan2(target_datas[i].v_y, target_datas[i].v);
    }
    target_datas[i].dtheta = planning_point_temp.Epsi();
    target_datas[i].t = i * mpc_during;
    targets_data_store.x = target_datas[i].x;
    targets_data_store.y = target_datas[i].y;
    targets_data_store.psi = target_datas[i].theta;
    // targets_data_store.psi = atan2(target_datas[i].y, target_datas[i].x);
    targets_data_store.epsi = target_datas[i].dtheta;
    targets_data_store.u = target_datas[i].v;
    targets_data_store.v = target_datas[i].y / ((i + 1) * mpc_during);
    printf("targets_data_store.v %f \n", targets_data_store.v);
    mpc_states_gl.targets.emplace_back(std::move(targets_data_store));
    kappa_mean += planning_point_temp.Kappa() / mpc_steps;
    acc_mean += Trajectory_->trajectory_points[i].a / mpc_steps;
  }

  elane::control::Config::instance().set_mean_kappa(kappa_mean);
  elane::control::Config::instance().set_mean_acceleration(acc_mean);

  // if (kappa_mean >= 0.01 && target_datas[mpc_steps].v > 1.5) {
  //   for (size_t i = 0; i < mpc_steps; i++) {
  //     target_datas[i].theta += kappa_gain_ * 0.2;
  //   }
  // }
}

void ControlNode::ThrottleToForce(double throttle, double &force) {
  for (size_t i = 1; i < 19; i++) {
    if (throttle <= (i * 5)) {
      force =
          force_spring[i - 1] + ((throttle - (i - 1) * 5) *
                                 (force_spring[i] - force_spring[i - 1]) / 5);

      printf("i %d \n", i);
      break;
    }
  }
}
}  // namespace control
}  // namespace elane