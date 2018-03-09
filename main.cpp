#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <opencv2/dnn/dnn.hpp>


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{ help h  | | show this message }"
        "{ video v | | (required) path to video }"
        "{ config c | | (required) path to configs file }"
        "{ weights w | | (required) path to weights file }"
        "{ class_names n | | (required) path to classnames file }"
    );

    if( parser.has("help") ){
        parser.printMessage();
        cerr << "" << endl;
    }

    string video_location(parser.get<string>("video"));
    if  (video_location.empty()){
        parser.printMessage();
        return -1;
    }

    VideoCapture video;
    video.open(video_location);

    if( !video.isOpened() ){
        cerr << "Could not open or find the video" << endl;
        return -1;
    }

    string modelConfiguration(parser.get<string>("config"));
    if  (modelConfiguration.empty()){
        cerr << "Could not open or find the configurations file" << endl;
        return -1;
    }

    string modelBinary(parser.get<string>("weights"));
    if  (modelBinary.empty()){
        cerr << "Could not open or find the weights file" << endl;
        return -1;
    }

    vector<string> classNamesVec;
    ifstream classNamesFile(parser.get<String>("class_names").c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    cv::dnn::experimental_dnn_v2::Net net = cv::dnn::experimental_dnn_v2::readNetFromDarknet(modelConfiguration, modelBinary);
    if( net.empty() ){
        cerr << "Can't load network" << endl;
        cerr << "cfg-file: " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        exit(-1);
    }
    Mat frame;

    while( true ){
        video >> frame;
        if( frame.empty() ){
            return 0;
        }
        ///imshow("Frame", frame);
        Mat inputBlob = cv::dnn::experimental_dnn_v2::blobFromImage(frame, 1/255.F, Size(416,416), Scalar(), true, false);
        net.setInput(inputBlob, "data");
        Mat detectionMat = net.forward("detection_out");

                vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;
        ostringstream ss;
        ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
        putText(frame, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));

        float confidenceThreshold = 0.24;
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
            if (confidence > confidenceThreshold)
            {
                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
                Rect object(xLeftBottom, yLeftBottom,
                            xRightTop - xLeftBottom,
                            yRightTop - yLeftBottom);
                rectangle(frame, object, Scalar(0, 255, 0));
                if (objectClass < classNamesVec.size())
                {
                    ss.str("");
                    ss << confidence;
                    String conf(ss.str());
                    String label = String(classNamesVec[objectClass]) + ": " + conf;
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom ),
                                          Size(labelSize.width, labelSize.height + baseLine)),
                              Scalar(255, 255, 255), CV_FILLED);
                    putText(frame, label, Point(xLeftBottom, yLeftBottom+labelSize.height),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
                }
                else
                {
                    cout << "Class: " << objectClass << endl;
                    cout << "Confidence: " << confidence << endl;
                    cout << " " << xLeftBottom
                         << " " << yLeftBottom
                         << " " << xRightTop
                         << " " << yRightTop << endl;
                }
            }
        }
        imshow("YOLO: Detections", frame);

        waitKey(1);

    }

    return 0;
}
