cmake_minimum_required(VERSION 3.10)
project(ship_control CXX)

# =============================================================================
# C++14 / 17 兼容（任务要求 C++14/17）
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wno-unused-variable -fPIC)

# =============================================================================
# catkin / ROS
# =============================================================================
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    elane_msgs
    message_generation
    tf
)
add_message_files(FILES ShieldIntervention.msg)
generate_messages(DEPENDENCIES std_msgs)

# =============================================================================
# 第三方依赖
#   * Eigen3 : 来自 apt 安装的 libeigen3-dev
#   * OSQP + osqp-eigen : Docker 镜像中源码编译并 install 到 /usr/local
#   * ONNX Runtime : /opt/onnxruntime (ENV ONNXRUNTIME_ROOT_PATH)
# =============================================================================
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

if(DEFINED ENV{ONNXRUNTIME_ROOT_PATH})
    set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT_PATH}")
else()
    set(ORT_ROOT "/opt/onnxruntime")
endif()
message(STATUS "ONNX Runtime root: ${ORT_ROOT}")

# =============================================================================
# catkin_package : 暴露 ShieldIntervention.h 给依赖包
# =============================================================================
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs elane_msgs message_runtime tf
    DEPENDS EIGEN3
)

# =============================================================================
# include / link 目录
# =============================================================================
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${ORT_ROOT}/include
)
link_directories(${ORT_ROOT}/lib)

# =============================================================================
# 主目标：motion_rl_shield_node
# =============================================================================
add_executable(motion_rl_shield_node
    src/motion_rl_shield.cpp
)
add_dependencies(motion_rl_shield_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(motion_rl_shield_node
    ${catkin_LIBRARIES}
    OsqpEigen::OsqpEigen
    onnxruntime          # 由 link_directories 指向 /opt/onnxruntime/lib
)
target_compile_definitions(motion_rl_shield_node PRIVATE
    ORT_API_MANUAL_INIT=0
)

install(TARGETS motion_rl_shield_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch config
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
