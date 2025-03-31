/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_HPP
#define RL_REAL_HPP

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "gamepad.hpp"
#include "depth_cleaner.hpp"
// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include <csignal>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::go2;
#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_IMU_TORSO "rt/secondary_imu"
// #define TOPIC_JOYSTICK "rt/wirelesscontroller"
#define TOPIC_LEFT_DEX3_CMD "rt/dex3/left/cmd"
#define TOPIC_LEFT_DEX3_STATE "rt/dex3/left/state"
#define TOPIC_RIGHT_DEX3_CMD "rt/dex3/right/cmd"
#define TOPIC_RIGHT_DEX3_STATE "rt/dex3/right/state"
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

using namespace unitree::common;

class RL_Real : public RL
{
public:
    RL_Real();
    ~RL_Real();

private:
    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();
    void ProcessDepth();

    // history buffer
    ObservationBuffer history_obs_buf;
    torch::Tensor history_obs;

    // depth
    DepthProcesser depth_cleaner;

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;
    std::shared_ptr<LoopFunc> loop_depth;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // unitree interface
    void InitMotionSwitcherClient();
    void InitLowCmd();
    uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
    void LowStateMessageHandler(const void *messages);
    void ImuTorsoHandler(const void *message);
    unitree::robot::b2::MotionSwitcherClient msc;
    unitree_hg::msg::dds_::LowCmd_ unitree_low_command{};
    unitree_hg::msg::dds_::LowState_ unitree_low_state{};
    unitree_hg::msg::dds_::IMUState_ unitree_imu_state{};
    REMOTE_DATA_RX unitree_rx;
    Gamepad unitree_gamepad;
    ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imu_subscriber;

    // others
    int motiontime = 0;
    std::vector<double> mapped_joint_positions;
    std::vector<double> mapped_joint_velocities;
    int command_mapping[29] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
    int state_mapping[29] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
};

#endif // RL_REAL_HPP
