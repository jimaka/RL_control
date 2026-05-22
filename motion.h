#pragma once
#include <math.h>

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

// #include "../ECL/ECL_L1_Pos_Controller.hpp"
#include "../Geo/geo/geo.h"
#include "../Pid/pid.hpp"
#include "../control_sim/control_sim.h"
#include "../mpc/data_store.hpp"
#include "../mpc/enu_car.h"
#include "../mpc/get_point.h"
#include "../mpc/mpc_AD.h"
#include "../mpc/mpc_rudder.h"
#include "../mpc/mpc_ta.h"
#include "../mpc/rls.h"
#include "../mpc/system_id.h"
#include "../mpc/xicheng_Azimuthing.h"
#include "/opt/elane/ros/include/botix_common/modules_results/modules_results_publisher.h"
#include "Eigen/Core"
#include "Eigen/QR"
#include "dynamic_reconfigure/server.h"
#include "elane_logger/elane_logger.h"
#include "elane_msgs/ControlCmd.h"
#include "elane_msgs/ControlDebug.h"
#include "elane_msgs/FusionPoseStamped.h"
#include "elane_msgs/Trajectory.h"
#include "elane_msgs/TrajectoryPoint.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/Polygon.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "matplot/matplot.h"
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Joy.h"
#include "sensor_msgs/NavSatFix.h"
#include "ship_control/GlobalPlannerConfig.h"
#include "ship_control/control_mpc_debug.h"
#include "ship_control/mpc_state.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int64.h"
#include "tf/tf.h"
#include "visualization_msgs/Marker.h"
#include "yaml-cpp/yaml.h"
// #include "elane_msgs/pa
#include "elane_msgs/ThrusterFeedback.h"
#include "los_pid.h"
#include "ship_hydro_model_pkg/MiddleHydroParams.h"

namespace elane {

namespace control {
class ControlNode {
 public:
  ControlNode(ros::NodeHandle &nh);

  ~ControlNode() = default;

  void Init();

  void YamlInit(std::string file_path);

  void SubTrajectory(const elane_msgs::Trajectory::ConstPtr &msg);

  void SubFusionPoseStamped(const elane_msgs::FusionPoseStamped::ConstPtr &msg);

  void SubUnderState(const elane_msgs::ControlCmd::ConstPtr &msg);

  void SubThrusterFeedback(const elane_msgs::ThrusterFeedback::ConstPtr &msg);

  void SubJoy(const sensor_msgs::Joy::ConstPtr &msg);

  void SubOnlinePara(
      const ship_hydro_model_pkg::MiddleHydroParams::ConstPtr &msg);

  void ShipBaseChange(const double target_x, const double target_y,
                      const double target_yaw, double &ship_base_x,
                      double &ship_base_y, double &ship_base_yaw);

  void MpcTaRunRudder();

  void MpcTaRunShunYi();

  void MpcTaRunXiCheng();

  void PublishControlCommand();

  void PublishDebugInfo();

  void Run(const ros::TimerEvent &time_e);

  void DisconnectJudge();

  void Stop();

  void ThrustReverse();

  void PointChange();

  void ThrottleToForce(double throttle, double &force);

 private:
  enum class CountId {
    NAV_COUNT = 0,
    PLAN_COUNT,
    DATA_RECORD_COUNT,
    MODULE_COUNT,
    ONLINE_SYS_ID,
    COUNT_ID_LENGTH

  };

  enum class SubscriberId {
    TRAJECTORY = 0,
    FUSIONPOSESTAMPED,
    JOY_MSG,
    UNDER_STATE,
    THRUSTER_FEEDBACK,
    ONLIE_PARA,
    SUBSCRIBER_ID_LENGTH
  };

  enum class PublisherId {
    WARNING_TOPIC = 0,
    CONTROL_CMD,
    DEBUG_MSG,
    TRACK_POINT_INDEX,
    PUBLISHER_ID_LENGTH

  };

  enum class ThrustModeId {
    DOUBLE_THRUST_SHUN_YI = 0,
    DOUBLE_THRUST_XI_CHENG,
    DOUBLE_RUDEEER,
    SINGLE_THRUST,
    SINGLE_RUDDER,
  };

  std::vector<ros::Subscriber> subs_;
  std::vector<ros::Publisher> pubs_;

  std::shared_ptr<botix::common::ModulesResultsPublisher>
      module_result_publisher_ = nullptr;
  elane_msgs::TrajectoryConstPtr Trajectory_ = nullptr;
  elane_msgs::FusionPoseStampedConstPtr FusionPoseStamped_ = nullptr;
  elane_msgs::ControlCmdConstPtr under_state_ = nullptr;
  elane_msgs::ControlCmdConstPtr output_msg_ = nullptr;
  elane_msgs::ThrusterFeedbackConstPtr thruster_feedback_ = nullptr;
  ship_hydro_model_pkg::MiddleHydroParamsConstPtr online_para_ = nullptr;
  ship_control::control_mpc_debug debug_msg_;
  elane::control::GetPoint get_point_;
  elane::control::PlanningPoint planning_point_;
  std::vector<elane::control::PlanningPoint> planning_points_;

  ros::Timer timer_;
  YAML::Node config_;
  int ship_id_ = 1;
  bool remote_flag_ = false;
  bool run_flag_ = false;
  bool data_record_flag_ = false;
  unsigned int count_[static_cast<int>(CountId::COUNT_ID_LENGTH)] = {0};

  xicheng_azimuthing::MpcTaRudderSlove xicheng_azimuthing_MpcTaRudderSlove_;
  MpcTaSlove MpcTaSlove_;
  elane::control_rls::Rls rls_;

  double force_tnn_ = 1310;
  double nav_t_ = 0;
  double nav_yaw_rate_;
  double nav_yaw_acc_;
  double x_acc_ = 0;
  double y_acc_ = 0;
  double yaw_rate_ = 0;
  double yaw_rate_old_ = 0;
  double yaw_rate_acc_ = 0;
  double x_v_ = 0;
  double y_v_ = 0;
  double yaw_ = 0;
  double rotation_center_x_ = 0;
  double rotation_center_y_ = 0;
  double rotation_center_distance_ = 0;
  double rotation_center_distance_lat_ = 0;

  std::vector<elane::control::PlanningPoint> planning_points_orin,
      planning_points_new;
  elane::control::PlanningPoint planning_point_temp;
  elane::control::GetPoint get_point;

  elane::control::LosPid los_pid_;
  double throttle_back_gain_ = 1.0;
  double kappa_gain_ = 0.5;
  double final_speed_ = 0;
  int ship_deadweight_ = 0;  // 0空载 1中载 2满载
};

}  // namespace control
}  // namespace elane