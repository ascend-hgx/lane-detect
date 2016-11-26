#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

#define PI 3.1415926

using namespace cv;
using namespace std;

int min_threshold = 50;
int max_trackbar = 150;

const string probabilistic_name = "Probabilistic Hough Lines Demo";

int sobelProcess(Mat grayImg, Mat& sobelImg);
int probabilisticHough(Mat img, int&  p_trackbar, void* data);

int main()
{
    int p_trackbar = 70;
//    string window_name = "Processed Video";
    namedWindow( probabilistic_name, WINDOW_AUTOSIZE );
//	namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
	VideoCapture capture("LaneDetection.avi");

	if(!capture.isOpened())
    {
        cout << "video open error" << endl;
        return -1;
    }

    double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << "Frame Size = " << dWidth << "x" << dHeight << endl;

    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
//    VideoWriter oVideoWriter ("LaneDetection.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true);

    /// Create Trackbars for Thresholds
    char thresh_label[50];
    sprintf( thresh_label, "Thres: %d + input", min_threshold );

    createTrackbar( thresh_label, probabilistic_name, &p_trackbar, max_trackbar);

   Mat frameImg;
   Mat grayImg;
   Mat sobelImg;
    while(1)
    {
        capture >> frameImg;
        if(frameImg.empty())
            break;
        cvtColor(frameImg, grayImg, CV_RGB2GRAY);
        imshow("gray image", grayImg);

        sobelProcess(grayImg, sobelImg);
        probabilisticHough(sobelImg, p_trackbar, 0);

        waitKey(10);
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}

int sobelProcess(Mat grayImg, Mat& sobelImg)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    if(grayImg.empty())
        return -1;
    //![reduce_noise]
    GaussianBlur( grayImg, grayImg, Size(3,3), 0, 0, BORDER_DEFAULT );

    //![sobel]
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( grayImg, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( grayImg, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    //![sobel]

    //![convert]
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    //![convert]

    //![blend]
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobelImg );
    //![blend]

    //![display]
    imshow( "sobel image", sobelImg );
}

int probabilisticHough(Mat img, int&  p_trackbar, void* data)
{
    vector<Vec4i> p_lines;
    Mat threshImg;

    threshold(img, threshImg, 88, 255, CV_THRESH_BINARY);
//    imshow("threshImg", threshImg);
    HoughLinesP( threshImg, p_lines, 1, CV_PI/180, min_threshold + p_trackbar, 30, 10 );

    for( size_t i = 0; i < p_lines.size(); i++ )
    {
        Vec4i l = p_lines[i];
        line( img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, LINE_AA);
    }

    // adaptive change p_trackbar value
    int s = p_lines.size();
    if(s > 10 &&
            p_trackbar <= (max_trackbar - 10))
        p_trackbar += 10;
    else if(s < 4 && p_trackbar >= 10)
        p_trackbar -= 10;

    cout << "p_trackbar value: " << p_trackbar << endl;
    cout << "line number: " << s << endl;

   imshow( probabilistic_name, img);
}
