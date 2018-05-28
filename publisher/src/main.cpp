#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <boost/assign/list_of.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/sync_queue.hpp>


using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    ros::NodeHandle _nh("~"); // to get the private params
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/videofile/image_raw", 1);
	
    sensor_msgs::ImagePtr msg;

	string video_location = "/home/wouter/Documenten/YOLO/Persoonsdetectie/people.mp4";
    if  (video_location.empty()){
        return -1;
    }

    VideoCapture video;
    video.open(video_location);

    if( !video.isOpened() ){
        cerr << "Could not open or find the video" << endl;
        return -1;
    }
	Mat frame;
	while( nh.ok() ){
		video >> frame;
		if( frame.empty() ){
			return 0;
		}
		msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
		pub.publish(msg);
		imshow("img", frame);
		waitKey(1);
		ros::spinOnce;	
	}
    return 0;
}
