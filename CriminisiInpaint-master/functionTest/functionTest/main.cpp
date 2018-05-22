#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;


void copyTo(Mat inputMat);
string type2str(int type);
void displayMat(string name, Mat mat);
Mat copyMat(Mat inputMat);

int main(int argc, const char * argv[]) {
    
    Mat src, src_gray, mask, mask_gray, grad_x, grad_y, inpaint_me;
    src = imread(argv[1]);  // CV_8UC3 339x225 => 225 rows, 339 cols
    mask = imread(argv[2]); // CV_8UC3 339x225 => 225 rows, 339 cols
    
    Mat copy = copyMat(src);
    displayMat("copy", copy);
    
    return 0;
}

// input = Mat that needs to be copied
// output = copied Mat
Mat copyMat(Mat inputMat) {
    // Mat (rows, cols)
    int rows = inputMat.rows;
    int cols = inputMat.cols;
    
    // declare and initialize output matrix
    // Mat.Size requires (cols,rows)! NOT (rows,cols)
    // Mat's constructor is Mat(rows,cols), but the Size(cols,rows) is flipped. Weird..
    Mat outputMat(cv::Size(cols, rows), CV_8UC3);
    outputMat = 0; // initialize the matrix with zeros
    
    // copy the pixel values
    for(int y=0; y<rows; y++)
    {
        for(int x=0; x<cols ;x++)
        {
            // get color (R,G,B) from the input matrix
            Vec3b inputColor = inputMat.at<Vec3b>(Point(x,y));
            //cout << "R: " << (int)inputColor[0] << endl;
            
            // copy to the output matrix
            outputMat.at<Vec3b>(Point(x,y)) = inputColor;
        }
    }
    return outputMat;
}

// usage:
// string ty =  type2str( M.type() );
// printf("Matrix: %s %dx%d \n", ty.c_str(), M.cols, M.rows );
string type2str(int type) {
    string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans+'0');
    
    return r;
}

// displays the Mat.
// input: window name, Mat object
void displayMat(string name, Mat mat) {
    imshow(name, mat);
    cv::waitKey(0);
}
