#include "depth_cleaner.hpp"

/*
* Awesome method for visualizing the 16bit unsigned depth data using a histogram, slighly modified (:
* Thanks to @Catree from https://stackoverflow.com/questions/42356562/realsense-opencv-depth-image-too-dark
*/
void DepthProcesser::make_depth_histogram(const Mat &depth, Mat &normalized_depth, int coloringMethod)
{
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

DepthProcesser::DepthProcesser(float near_clip_m, float far_clip_m)
  :near_clip(near_clip_m * 1000.0f), far_clip(far_clip_m * 1000.0f)
{
  //Create a depth cleaner instance
  depthc = new rgbd::DepthCleaner(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

  //Add desired streams to configuration
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

  // Start streaming with default recommended configuration
  pipe.start(cfg); 

  // openCV window
  namedWindow(window_name_source, WINDOW_AUTOSIZE);
  namedWindow(window_name_filter, WINDOW_AUTOSIZE);
}

void DepthProcesser::process_depth()
{
  std::cout << "test" << std::endl;
  data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
  depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
  std::cout << "test" << std::endl;
  if (!depth_frame) // Should not happen but if the pipeline is configured differently
      return;       //  it might not provide depth and we don't want to crash

  std::cout << "good" << std::endl;

  // Query frame size (width and height)
  const int w = depth_frame.as<rs2::video_frame>().get_width();
  const int h = depth_frame.as<rs2::video_frame>().get_height();

  // //Create queued mat containers
  // QueuedMat depthQueueMat;
  // QueuedMat cleanDepthQueueMat;

  // Create an openCV matrix from the raw depth (CV_16U holds a matrix of 16bit unsigned ints)
  Mat rawDepthMat(Size(w, h), CV_16U, (void*)depth_frame.get_data());
  std::cout << rawDepthMat.at<uint16_t>(w/2, h/2) << std::endl;

  std::cout << "rawDepthMat type: " << rawDepthMat.type() << std::endl;

  // // Create an openCV matrix for the DepthCleaner instance to write the output to
  // Mat cleanedDepth(Size(w, h), CV_16U);

  // //Run the RGBD depth cleaner instance
  // depthc->operator()(rawDepthMat, cleanedDepth);

  // const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
  // Mat temp, temp2;

  // // // Downsize for performance, use a smaller version of depth image (defined in the SCALE_FACTOR macro)
  // // Mat small_depthf;
  // // resize(cleanedDepth, small_depthf, Size(), SCALE_FACTOR, SCALE_FACTOR);

  // // Inpaint only the masked "unknown" pixels
  // inpaint(cleanedDepth, (cleanedDepth == noDepth), temp, 5.0, INPAINT_TELEA);

  // // Upscale to original size and replace inpainted regions in original depth image
  // resize(temp, temp2, cleanedDepth.size());
  // temp2.copyTo(cleanedDepth, (cleanedDepth == noDepth));  // add to the original signal

  // // threshold
  // Mat temp_thresholded, temp2_thresholded;
  // threshold(cleanedDepth, temp_thresholded, far_clip, far_clip, THRESH_TRUNC);
  // threshold(temp_thresholded, temp2_thresholded, near_clip, near_clip, THRESH_TOZERO);

  // cleanedDepth = temp2_thresholded.clone();

  // // resize

  // // crop

  // // // normalize
  // // Mat cleanedDepth_float;
  // // cleanedDepth.convertTo(cleanedDepth_float, CV_32F);

  // // Mat normalized_depth = (cleanedDepth_float - near_clip) / (far_clip - near_clip);

  // imshow(window_name_filter, cleanedDepth);
  imshow(window_name_source, rawDepthMat);
}

// int main(int argc, char * argv[]) try {

//     // The threaded processing thread function
//     std::thread processing_thread([&]() {
//         while (!stopped){
//             // threshold
//             Mat temp_thresholded, temp2_thresholded;
//             threshold(cleanedDepth, temp_thresholded, far_clip, far_clip, THRESH_TRUNC);
//             threshold(temp_thresholded, temp2_thresholded, near_clip, near_clip, THRESH_TOZERO);

//             cleanedDepth = temp2_thresholded.clone();

//             // normalize
//             Mat cleanedDepth_float;
//             cleanedDepth.convertTo(cleanedDepth_float, CV_32F);

//             Mat normalized_depth = (cleanedDepth_float - near_clip) / (far_clip - near_clip);

//             cv::Mat vis_image;
//             normalized_depth.convertTo(vis_image, CV_8U, 255.0);

//             // Use the copy constructor to copy the cleaned mat if the isDepthCleaning is true
//             cleanDepthQueueMat.img = cleanedDepth;

//             //Use the copy constructor to fill the original depth coming in from the sensr(i.e visualized in RGB 8bit ints)
//             depthQueueMat.img = rawDepthMat;

//             //Push the mats to the queue
//             originalQueue.push(depthQueueMat);
//             filteredQueue.push(cleanDepthQueueMat);

//             imshow(window_name_filter, vis_image);
//             // imshow(window_name_filter, cleanedDepth);
//             imshow(window_name_source, rawDepthMat);
//         }
//     });

//     Mat filteredDequeuedMat(Size(1280, 720), CV_16UC1);
//     Mat originalDequeuedMat(Size(1280, 720), CV_8UC3);

//     //Main thread function
//     while (waitKey(1) < 0 && cvGetWindowHandle(window_name_source) && cvGetWindowHandle(window_name_filter)){

//         //If the frame queue is not empty pull a frame out and clean the queue
//         while(!originalQueue.empty()){
//             originalQueue.front().img.copyTo(originalDequeuedMat);
//             originalQueue.pop();
//         }


//         while(!filteredQueue.empty()){
//             filteredQueue.front().img.copyTo(filteredDequeuedMat);
//             filteredQueue.pop();
//         }

//         Mat coloredCleanedDepth;
//         Mat coloredOriginalDepth;

//         make_depth_histogram(filteredDequeuedMat, coloredCleanedDepth, COLORMAP_JET);
//         make_depth_histogram(originalDequeuedMat, coloredOriginalDepth, COLORMAP_JET);

//         // imshow(window_name_filter, coloredCleanedDepth);
//         // imshow(window_name_source, coloredOriginalDepth);
//         // imshow(window_name_filter, filteredDequeuedMat);
//         // imshow(window_name_source, originalDequeuedMat);
//     }

//     // Signal the processing thread to stop, and join
//     stopped = true;
//     processing_thread.join();

//     return EXIT_SUCCESS;
// }
