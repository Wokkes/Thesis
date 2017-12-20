#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main( int argc, char** argv )
{
    String image_location;
    String path = "/home/wouter/Documenten/HOG/HOGtest/Dataset/person_";
    String extensie = ".png";
    char teller[3];
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<double> tijden;

    for( int i = 7; i <= 350; i++ ){
        ///padnaam correct opbouwen
        sprintf(teller, "%.3d", i);
        ostringstream convert;
        convert << teller;
        image_location = path + convert.str( )+ extensie;
        cout << image_location << endl;

        ///Afbeelding inlezen
        Mat img = imread(image_location);
        if( img.empty() ){
            cerr << "Could not open or find the image" << endl;
            continue;
            ///Overslaan van niet bestaande afbeeldingen
        }

        ///Detectie van personen
        vector<Rect> people;
        vector<double> weights;

        double t = (double) getTickCount();
        hog.detectMultiScale(img, people, weights, 0.75, Size(8,8), Size(0,0), 1.05, 2);
        t = (double)getTickCount() - t;
        cout << "Detection time: " << (t*1000/getTickFrequency()) << " ms" << endl;
        tijden.push_back(t*1000/getTickFrequency());

        Mat canvas = img.clone();

        ///Personen tekenen
        for( size_t j = 0; j < people.size(); j++ ){
            rectangle(canvas, people[j], Scalar(0, 0, 255));
            stringstream temp;
            temp << (double)weights[j];
            putText(canvas, temp.str(), Point(people[j].x, people[j].y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
        }

        ///Resultaten weergeven
        ///imshow("Detecties", canvas);
        ///waitKey(0);

    }

    ///Berekenen van gemiddelde detectietijd
    double tijd = 0.0;
    for(size_t i = 0; i < tijden.size(); i++){
        tijd += tijden[i];
    }
    tijd = tijd/tijden.size();
    cout << "Aantal verwerkte afbeeldingen: " << tijden.size() << endl;
    cout << "Gemiddelde detectietijd: " << tijd << " ms" << endl;

    return 0;
}
