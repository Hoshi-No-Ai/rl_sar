/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_g1.hpp"

// #define PLOT
// #define CSV_LOGGER

bool if_gpu = false;

RL_Real::RL_Real()
{
    // read params from yaml
    this->robot_name = "g1_isaacgym";
    this->ReadYaml(this->robot_name);
    for (std::string &observation : this->params.observations)
    {
        // In Unitree G1, the coordinate system for angular velocity is in the body coordinate system.
        if (observation == "ang_vel")
        {
            observation = "ang_vel_body";
        }
    }

    // init robot
    this->InitMotionSwitcherClient();
    std::string form, name;
    while (this->msc.CheckMode(form, name), !name.empty()) {
      if (this->msc.ReleaseMode())
        std::cout << "Failed to switch to Release Mode\n";
      sleep(5);
    }
    this->InitLowCmd();
    // create publisher
    this->lowcmd_publisher.reset(new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    this->lowcmd_publisher->InitChannel();
    // create subscriber
    this->lowstate_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    this->lowstate_subscriber->InitChannel(std::bind(&RL_Real::LowStateMessageHandler, this, std::placeholders::_1), 1);

    // this->imu_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>(TOPIC_IMU_TORSO));
    // this->imu_subscriber->InitChannel(std::bind(&RL_Real::ImuTorsoHandler, this, std::placeholders::_1), 1);

    // init rl
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size());
    }
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
    running_state = STATE_ZERO_TORQUE;

    this->device_type = at::kCPU; // 定义设备类型
    if (torch::cuda::is_available() && if_gpu)
    {
        this->device_type = at::kCUDA;
        std::cout << "CUDA is available! Running on the GPU!" << std::endl;
    }

    // model
    std::cout << "if CUDA available: " << torch::cuda::is_available() << std::endl;
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + this->robot_name + "/" + this->params.model_name;
    this->model = torch::jit::load(model_path);
    this->model.to(this->device_type);

    // // depth
    // depth_cleaner = DepthProcesser(0.3,3.0);

    // loop
    // this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    // this->loop_depth = std::make_shared<LoopFunc>("loop_depth", this->params.dt * this->params.decimation, std::bind(&RL_Real::ProcessDepth, this));
    // this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
    // this->loop_depth->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    // this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<double> *state)
{
    if ((int)this->unitree_gamepad.A.pressed == 1)
    {
        this->control.control_state = STATE_MOVING_DEFAULT_POS;
    }
    else if ((int)this->unitree_gamepad.B.pressed == 1)
    {
        this->control.control_state = STATE_RL_INIT;
    }
    else if ((int)this->unitree_gamepad.X.pressed == 1)
    {
        this->control.control_state = STATE_ZERO_TORQUE;
    }
    else if ((int)this->unitree_gamepad.select.pressed == 1)
    {
        this->control.control_state = STATE_DAMPING;
    }

    if (this->params.framework == "isaacgym")
    {
        state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[0]; // w
        state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[1]; // x
        state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[2]; // y
        state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[3]; // z
    }
    else if (this->params.framework == "isaacsim")
    {
        state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[0]; // w
        state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[1]; // x
        state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[2]; // y
        state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[3]; // z
    }

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }
    for (int i = 0; i < this->params.num_of_dofs + this->params.num_of_arm_waist_dofs; ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state()[state_mapping[i]].q();
        state->motor_state.dq[i] = this->unitree_low_state.motor_state()[state_mapping[i]].dq();
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state()[state_mapping[i]].tau_est();
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs + this->params.num_of_arm_waist_dofs; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = 0x01;
        this->unitree_low_command.motor_cmd()[i].q() = command->motor_command.q[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].dq() = command->motor_command.dq[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].kp() = command->motor_command.kp[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].kd() = command->motor_command.kd[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].tau() = command->motor_command.tau[command_mapping[i]];
    }

    this->unitree_low_command.crc() = Crc32Core((uint32_t *)&unitree_low_command, (sizeof(unitree_hg::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(unitree_low_command);
}

void RL_Real::RobotControl()
{
    this->motiontime++;

    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    // this->SetCommand(&this->robot_command);
}

void RL_Real::ProcessDepth()
{
    this->depth_cleaner.process_depth();
}

void RL_Real::RunModel()
{
    if (this->running_state == STATE_RL_RUNNING)
    {
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        this->obs.commands = torch::tensor({{this->unitree_gamepad.ly, -this->unitree_gamepad.lx, -this->unitree_gamepad.rx}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        // std::cout << "----------------------------------------------------" << std::endl;
        // std::cout << "ang_vel: " << this->obs.ang_vel << std::endl;
        // std::cout << "commands: " << this->obs.commands << std::endl;
        // std::cout << "base_quat: " << this->obs.base_quat << std::endl;
        // std::cout << "dof_pos: " << this->obs.dof_pos << std::endl;
        // std::cout << "dof_vel: " << this->obs.dof_vel << std::endl;
        // std::cout << "gravity_vec: " << this->QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec, this->params.framework) << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        torch::Tensor clamped_actions = this->Forward().to(at::kCPU);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Execution time: " << 1000.0f * duration.count() << " ms" << std::endl;

        this->obs.actions = clamped_actions;

        for (int i : this->params.hip_scale_reduction_indices)
        {
            clamped_actions[0][i] *= this->params.hip_scale_reduction;
        }

        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Real::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation().to(this->device_type);
    // std::cout << "clamped_obs Tensor device: " << clamped_obs.device() << std::endl;

    torch::Tensor actions;
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history).to(this->device_type);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower.to(this->device_type), this->params.clip_actions_upper.to(this->device_type));
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state()[i].q());
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd()[i].q());
        plt::subplot(4, 3, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

uint32_t RL_Real::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real::InitLowCmd()
{
    this->unitree_low_command.mode_pr() = static_cast<uint8_t>(Mode::PR);
    for (int i = 0; i < this->params.num_of_dofs + this->params.num_of_arm_waist_dofs; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        this->unitree_low_command.motor_cmd()[i].q() = (0);
        this->unitree_low_command.motor_cmd()[i].kp() = (0);
        this->unitree_low_command.motor_cmd()[i].dq() = (0);
        this->unitree_low_command.motor_cmd()[i].kd() = (0);
        this->unitree_low_command.motor_cmd()[i].tau() = (0);
    }
}

void RL_Real::InitMotionSwitcherClient()
{
    this->msc.SetTimeout(10.0f);
    this->msc.Init();
}

void RL_Real::LowStateMessageHandler(const void *message)
{
    this->unitree_low_state = *(unitree_hg::msg::dds_::LowState_ *)message;
    memcpy(this->unitree_rx.buff, &this->unitree_low_state.wireless_remote()[0], 40);
    this->unitree_gamepad.update(this->unitree_rx.RF_RX);
    // std::cout << "LowStateMessageHandler" << std::endl;

    if (this->unitree_low_command.mode_machine() != unitree_low_state.mode_machine()) {
      if (this->unitree_low_command.mode_machine() == 0) std::cout << "G1 type: " << unsigned(unitree_low_state.mode_machine()) << std::endl;
      this->unitree_low_command.mode_machine() = unitree_low_state.mode_machine();
    }
}

void RL_Real::ImuTorsoHandler(const void *message)
{
    this->unitree_imu_state = *(unitree_hg::msg::dds_::IMUState_ *)message;
    // std::cout << "ImuTorsoHandler" << std::endl;
}

void signalHandler(int signum)
{
    exit(0);
}
void make_depth_histogram(const Mat &depth, Mat &normalized_depth, int coloringMethod) {
    normalized_depth = Mat(depth.size(), CV_8U);
    int width = depth.cols, height = depth.rows;
  
    static uint32_t histogram[0x10000];
    memset(histogram, 0, sizeof(histogram));
  
    for(int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        ++histogram[depth.at<ushort>(i,j)];
      }
    }
  
    for(int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i-1]; // Build a cumulative histogram for the indices in [1,0xFFFF]
  
    for(int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        if (uint16_t d = depth.at<ushort>(i,j)) {
          int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
          normalized_depth.at<uchar>(i,j) = static_cast<uchar>(f);
        } else {
          normalized_depth.at<uchar>(i,j) = 0;
        }
      }
    }
  
    // Apply the colormap:
    applyColorMap(normalized_depth, normalized_depth, coloringMethod);
  }

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    ChannelFactory::Instance()->Init(0, argv[1]);

    RL_Real rl_sar;

     //Create a depth cleaner instance
     rgbd::DepthCleaner* depthc = new rgbd::DepthCleaner(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

     // A librealsense class for mapping raw depth into RGB (pretty visuals, yay)
     rs2::colorizer color_map;
 
     float near_clip = 300.0f; // 0.3m
     float far_clip = 3000.0f; // 3m
 
     // Declare RealSense pipeline, encapsulating the actual device and sensors
     rs2::pipeline pipe;
 
     //Create a configuration for configuring the pipeline with a non default profile
     rs2::config cfg;
 
     //Add desired streams to configuration
     cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
 
     // Start streaming with default recommended configuration
     pipe.start(cfg);
 
     // openCV window
     const auto window_name_source = "Source Depth";
     namedWindow(window_name_source, WINDOW_AUTOSIZE);
 
     const auto window_name_filter = "Filtered Depth";
     namedWindow(window_name_filter, WINDOW_AUTOSIZE);
 
     // Atomic boolean to allow thread safe way to stop the thread
     std::atomic_bool stopped(false);
 
     // Declaring two concurrent queues that will be used to push and pop frames from different threads
     std::queue<QueuedMat> filteredQueue;
     std::queue<QueuedMat> originalQueue;
 
     // The threaded processing thread function
     std::thread processing_thread([&]() {
         while (!stopped){
 
             rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
             rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
             if (!depth_frame) // Should not happen but if the pipeline is configured differently
                 return;       //  it might not provide depth and we don't want to crash
 
             //Save a reference
             rs2::frame filtered = depth_frame;
 
             // Query frame size (width and height)
             const int w = depth_frame.as<rs2::video_frame>().get_width();
             const int h = depth_frame.as<rs2::video_frame>().get_height();
 
             //Create queued mat containers
             QueuedMat depthQueueMat;
             QueuedMat cleanDepthQueueMat;
 
             // Create an openCV matrix from the raw depth (CV_16U holds a matrix of 16bit unsigned ints)
             Mat rawDepthMat(Size(w, h), CV_16U, (void*)depth_frame.get_data());
 
             // std::cout << rawDepthMat.at<uint16_t>(w/2, h/2) << std::endl;
 
             // Create an openCV matrix for the DepthCleaner instance to write the output to
             Mat cleanedDepth(Size(w, h), CV_16U);
 
             //Run the RGBD depth cleaner instance
             depthc->operator()(rawDepthMat, cleanedDepth);
 
             const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
             Mat temp, temp2;
 
             // Downsize for performance, use a smaller version of depth image (defined in the SCALE_FACTOR macro)
             Mat small_depthf;
             resize(cleanedDepth, small_depthf, Size(), SCALE_FACTOR, SCALE_FACTOR);
 
             // Inpaint only the masked "unknown" pixels
             inpaint(small_depthf, (small_depthf == noDepth), temp, 5.0, INPAINT_TELEA);
 
             // Upscale to original size and replace inpainted regions in original depth image
             resize(temp, temp2, cleanedDepth.size());
             temp2.copyTo(cleanedDepth, (cleanedDepth == noDepth));  // add to the original signal
 
             // threshold
             Mat temp_thresholded, temp2_thresholded;
             threshold(cleanedDepth, temp_thresholded, far_clip, far_clip, THRESH_TRUNC);
             threshold(temp_thresholded, temp2_thresholded, near_clip, near_clip, THRESH_TOZERO);
 
             cleanedDepth = temp2_thresholded.clone();
 
             // normalize
             Mat cleanedDepth_float;
             cleanedDepth.convertTo(cleanedDepth_float, CV_32F);
 
             Mat normalized_depth = (cleanedDepth_float - near_clip) / (far_clip - near_clip);
 
             cv::Mat vis_image;
             normalized_depth.convertTo(vis_image, CV_8U, 255.0);
 
             // Use the copy constructor to copy the cleaned mat if the isDepthCleaning is true
             cleanDepthQueueMat.img = cleanedDepth;
 
             //Use the copy constructor to fill the original depth coming in from the sensr(i.e visualized in RGB 8bit ints)
             depthQueueMat.img = rawDepthMat;
 
             //Push the mats to the queue
             originalQueue.push(depthQueueMat);
             filteredQueue.push(cleanDepthQueueMat);
 
             imshow(window_name_filter, vis_image);
             // imshow(window_name_filter, cleanedDepth);
             imshow(window_name_source, rawDepthMat);
         }
     });
 
     Mat filteredDequeuedMat(Size(1280, 720), CV_16UC1);
     Mat originalDequeuedMat(Size(1280, 720), CV_8UC3);
 
     //Main thread function
     while (waitKey(1) < 0 && cvGetWindowHandle(window_name_source) && cvGetWindowHandle(window_name_filter)){
 
         //If the frame queue is not empty pull a frame out and clean the queue
         while(!originalQueue.empty()){
             originalQueue.front().img.copyTo(originalDequeuedMat);
             originalQueue.pop();
         }
 
 
         while(!filteredQueue.empty()){
             filteredQueue.front().img.copyTo(filteredDequeuedMat);
             filteredQueue.pop();
         }
 
         Mat coloredCleanedDepth;
         Mat coloredOriginalDepth;
 
         make_depth_histogram(filteredDequeuedMat, coloredCleanedDepth, COLORMAP_JET);
         make_depth_histogram(originalDequeuedMat, coloredOriginalDepth, COLORMAP_JET);
 
         // imshow(window_name_filter, coloredCleanedDepth);
         // imshow(window_name_source, coloredOriginalDepth);
         // imshow(window_name_filter, filteredDequeuedMat);
         // imshow(window_name_source, originalDequeuedMat);
     }
 
     // Signal the processing thread to stop, and join
     stopped = true;
     processing_thread.join();

    while (1)
    {
        sleep(10);
    }

    // const char* window_name_source = "Source Depth";
    // const char* window_name_filter = "Filtered Depth";

    // while (waitKey(1) < 0 && cvGetWindowHandle(window_name_source))
    // {
        
    // }

    return 0;
}
