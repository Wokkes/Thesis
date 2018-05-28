#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <opencv2/dnn/dnn.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "persoonsdetectie/personen.h"


using namespace std;
using namespace cv;

Mat frame;
KalmanFilter KF(4, 2, 0);
Mat state = Mat::zeros(4, 2, CV_32F); 
Mat processNoise(4, 2, CV_32F);
Mat measurement = Mat::zeros(2, 1, CV_32F);
vector<Point> prevPts;
Point estPt = Point(0.0,0.0);
vector<string> classNamesVec;
cv::dnn::experimental_dnn_v2::Net net;
vector<Point> personen;
int maxcount = 10;
int counter = 0;
bool detection = false;

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
	try{
		Point persoon;
		personen.clear();
		frame = cv_bridge::toCvShare(msg,"bgr8")->image.clone();
		imshow("view", frame);
		waitKey(1);
		///Setup Neural Network
        Mat inputBlob = cv::dnn::experimental_dnn_v2::blobFromImage(frame, 1/255.F, Size(416,416), Scalar(), true, false);
        net.setInput(inputBlob, "data");
        Mat detectionMat = net.forward("detection_out");

        float confidenceThreshold = 0.75;
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
            if (confidence > confidenceThreshold)
            {
                if (objectClass == 0)
                {
					counter = 0;
					detection = true;
                    ///Get coordinates and properties of detected persons
                    float x = detectionMat.at<float>(i, 0);
                    float y = detectionMat.at<float>(i, 1);
                    float width = detectionMat.at<float>(i, 2);
                    float height = detectionMat.at<float>(i, 3);

					persoon.x = x*frame.cols;
					persoon.y = y*frame.rows;
					personen.push_back(persoon);

                    ///Get measurements
                    measurement.at<float>(0) = x*frame.cols;
                    measurement.at<float>(1) = y*frame.rows;
                    ///Update
                    state = KF.correct(measurement);
                    estPt.x = state.at<float>(0);
                    estPt.y = state.at<float>(1);
					cout << "update " << estPt << endl;
					///Draw tracking line
                    prevPts.push_back(estPt);
                    for(size_t j = 0; j < prevPts.size()-1; j++){
                        line(frame, prevPts[j], prevPts[j+1], Scalar(255,100,150),2);
                    }

                    ///Draw rectangle
                    int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                    int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                    int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                    int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
                    Rect object(xLeftBottom, yLeftBottom,
                                xRightTop - xLeftBottom,
                                yRightTop - yLeftBottom);
                    rectangle(frame, object, Scalar(0, 0, 255), 5);

                    cout << String(classNamesVec[objectClass]) << " at x = " << x << " y = " << y << endl;

					///Predict next state
                    state = KF.predict();
                    estPt.x = state.at<float>(0);
					estPt.y = state.at<float>(1);
					cout << "predict " << estPt << endl;
                    
                }
            }
		}		
		if (! detection ){
			if(counter < maxcount){
				counter++;
				prevPts.push_back(estPt);
				personen.push_back(estPt);
			}
			else{
				prevPts.clear();
				state = Mat::zeros(4,2,CV_32F);
			}
		}	
		imshow("YOLO: Detections", frame);
        waitKey(1);

		cout << "Image received" << endl;
		cout << prevPts << endl;
		detection = false;
	}
	catch (cv_bridge::Exception& e){
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}

int main(int argc, char** argv)
{
	/// Init ROS & subscribe to an image
	ros::init(argc, argv, "Persoonsdetector");
	ros::NodeHandle nh;
	ros::Publisher pub = nh.advertise<persoonsdetectie::personen>("personen", 1); 
	image_transport::ImageTransport it(nh);
	image_transport::Subscriber sub = it.subscribe("videofile/image_raw", 1000, imageCallback);
	persoonsdetectie::personen msg;
	///Load cfg, weights and class-names files.
	string modelConfiguration = "/home/wouter/darknet/cfg/yolo.cfg";
	if  (modelConfiguration.empty()){
		cerr << "Could not open or find the configurations file" << endl;
		return -1;
	}

	string modelBinary = "/home/wouter/darknet/yolo.weights";
	if  (modelBinary.empty()){
		cerr << "Could not open or find the weights file" << endl;
		return -1;
	}

	ifstream classNamesFile("/home/wouter/darknet/data/coco.names");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
		    classNamesVec.push_back(className);
	}

	net = cv::dnn::experimental_dnn_v2::readNetFromDarknet(modelConfiguration, modelBinary);
	if( net.empty() ){
		cerr << "Can't load network" << endl;
		cerr << "cfg-file: " << modelConfiguration << endl;
		cerr << "weights-file: " << modelBinary << endl;
		exit(-1);
	}

	/// Setup Kalman filter
	measurement.setTo(Scalar(0));
	KF.transitionMatrix = (Mat_<float>(4,4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(1));
	KF.statePre.at<float>(0) = 0;
	KF.statePre.at<float>(1) = 0;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;

	while( nh.ok() ){
		for(int i = 0; i < personen.size(); i++){
			geometry_msgs::Point point;
			point.x = personen[i].x;
			point.y = personen[i].y;
			msg.points.push_back(point);
		}
		pub.publish(msg);
		ros::spinOnce();
    }

    return 0;
}
