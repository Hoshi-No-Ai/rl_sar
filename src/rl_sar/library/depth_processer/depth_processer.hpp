#ifndef DEPTH_CLEANER_HPP
#define DEPTH_CLEANER_HPP

// Libraries
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/rgbd.hpp>     // OpenCV RGBD Contrib package
#include <opencv2/highgui/highgui_c.h> // OpenCV High-level GUI

// STD
#include <string>
#include <thread>
#include <atomic>
#include <queue>

using namespace cv;

#define SCALE_FACTOR 1

/*
* Class for enqueuing and dequeuing cv::Mats efficiently
* Thanks to this awesome post by PKLab
* http://pklab.net/index.php?id=394&lang=EN
*/
class QueuedMat
{
public:
    Mat img; // Standard cv::Mat

    QueuedMat(){}; // Default constructor

    // Destructor (called by queue::pop)
    ~QueuedMat(){
        img.release();
    };

    // Copy constructor (called by queue::push)
    QueuedMat(const QueuedMat& src){
        src.img.copyTo(img);
    };
};

class DepthProcesser
{
public:
    DepthProcesser(){};
    DepthProcesser(float near_clip_m, float far_clip_m);
    ~DepthProcesser(){};

    void process_depth();

    void make_depth_histogram(const Mat &depth, Mat &normalized_depth, int coloringMethod);

private:
    rgbd::DepthCleaner depthc;
    float near_clip;
    float far_clip;
    rs2::pipeline pipe;
    rs2::config cfg;
    std::queue<QueuedMat> filteredQueue;
    std::queue<QueuedMat> originalQueue;

    const char* window_name_source = "Source Depth";
    const char* window_name_filter = "Filtered Depth";

    rs2::frameset data;
    rs2::frame depth_frame;
};

#endif // DEPTH_CLEANER_HPP