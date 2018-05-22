#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

int win_size = 9;
int stride = 3;

void showMatBasics(String srcName, String maskName);
// https://stackoverflow.com/questions/6302171/convert-uchar-mat-to-float-mat-in-opencv
// https://stackoverflow.com/questions/13484475/what-are-the-upper-and-lower-limits-and-types-of-pixel-values-in-opencv

/*
 * Basic stuff about Mat
 * Mat (Row, Col)
 */


int main(int argc, char* argv[])
{
    showMatBasics(argv[1], argv[2]);
    
    if(argc < 3)
    {
        printf("Usage: %s src mask output\n",argv[0]);
        return -1;
    }
    
    Mat src, src_gray, mask, mask_gray, grad_x, grad_y, inpaint_me;
    
    // read source and mask(target) image
    src = imread(argv[1]);
    mask = imread(argv[2]);
    
    if(!src.data || !mask.data){return -1;}
    //Experimental resize!
    //    resize(src, src, Size(0,0), 0.5, 0.5);
    //    resize(mask, mask, Size(0,0), 0.5, 0.5);
    
    // display source image
    // imshow( "Display window", src );
    //cv::waitKey(0);
    
    // display mask(target) image
    // imshow( "Display window", mask );
    //cv::waitKey(0);
    
    // copy mask(target) to inpaint_me which is a BGR matrix
    src.copyTo(inpaint_me, mask);
    
    // https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    // L – Lightness ( Intensity ).
    // a – color component ranging from Green to Magenta.
    // b – color component ranging from Blue to Yellow.
    cvtColor(inpaint_me, inpaint_me, CV_BGR2Lab);//inpaint_me is now in Lab space
    
    // Computing gradients
    // Smoothing the source image, I don't know why
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT);
    // make src_Gray which is grayscale image
    cvtColor(src, src_gray, CV_BGR2GRAY);
    // make mask(target)_gray which is grayscal image
    cvtColor(mask, mask_gray, CV_BGR2GRAY);
    // 32FC1 means that each pixel value is stored as one channel floating point with single precision
    // ex) 0 -> 0.0, 24 -> 24.0, 255 -> 255.0
    // CV is unsigned by default
    // https://livierickson.com/blog/why-do-we-store-color-values-the-way-we-do/
    // [0 - 255] can be represented as [0. - 1.] floating points
    // [0 - 255] becomes [0.0 - 1.0] because it's rescaled[1./255.]
    Mat src_gray_f; // source gray floating-point
    src_gray.convertTo(src_gray_f, CV_32FC1, 1./255.);
    
    // grad_x is a derivative of src_gray.
    // it shows sudden changes in x-direction
    // grad_y shows sundden changes in y-direction
    Sobel(src_gray_f, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(src_gray_f, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    //imshow( "grey", src_gray);
    //imshow( "sobel  x", grad_x);
    //imshow( "sobel y", grad_y );
    //cv::waitKey(0);
    
    // create patches inside I - \omega, and find boundary
    // patches si matrix, each row is a single patch vector (casted to float)
    int num_patches = 0;
    
    // mask rows = 255
    // mask cols = 339
    // win_size = 9
    // packing the target image with the patches = (339 - 9) * (225 - 9) = 71280
    // -9 indicates so that the window does not need extra padding
    // however, becasue of the stride, (71280 / 9) = 7920
    // so total, 7920 patches are inside Source Region!, not in the target
    // howver, author says inside I - omega(target), which is SOURCE, so I am not sure...
    // now it makes sense, go to the for loops. Patches are inside I - omega(target)!!!
    int num_vec = (mask.rows - win_size) * (mask.cols - win_size) / (stride * stride);
    // f_vec_len indicates the windows dimension depending on the number of channels
    int f_vec_len = (win_size * win_size) * src.channels();
    // 7920 patches(rows). Each row is a patch which is the window size (9 * 9 * numChannel)
    Mat patches(num_vec, f_vec_len, CV_8UC1);
    
    // iterating mask(target)
    // Matrix is Mat (Y, X) not Mat( X, Y)
    // cout << "mask y: " << mask_gray.rows << endl;
    // cout << "mask x: " << mask_gray.cols << endl;
    // Mat test = mask_gray(Range(0, 225), Range(0, 165)).clone();
    // imshow("test clone", test);
    // cv::waitKey(0);
    
    // inpaint_me = copy of mask(target) and converted to Lab, not BGR.
    /*
     Mat maskCopy = mask(Range(0,225), Range(0,339)).clone();
     Mat inpaintCopy = inpaint_me(Range(0,225), Range(0,339)).clone();
     Mat inpaintReshape = inpaint_me(Range(0,225), Range(0,339)).clone().reshape(1,3);
     
     imshow("mask", maskCopy );
     imshow("inpaint (Lab)", inpaintCopy);
     imshow("source", src );
     imshow("inpaint rehspae(1,1)", inpaintReshape);
     cv::waitKey(0);
     */
    
    /*
     Mat nonZeroMat = mask_gray(Range(0,20), Range(0,20));
     imshow("countNonZero ", nonZeroMat);
     cv::waitKey(0);
     // it returns 400 because white spaces are all non-zeros
     cout << "top left mask gray # of nonzeros: " << countNonZero(nonZeroMat) << endl;
     */
    
    /*
     // Create patches in Image - omega(target or mask)
     for(int y =0; y < mask.rows - win_size; y+=stride)
     {
     for(int x =0; x < mask.cols - win_size; x+=stride)
     {
     // create patch
     // get the patch of inpaint_me (Lab space)
     Mat patch_mat = inpaint_me(Range(y,y+win_size), Range(x,x+win_size)).clone();
     // get the patch of mask(target)
     Mat mask_patch = mask_gray(Range(y,y+win_size), Range(x,x+win_size)).clone();
     // reshape the mask(target) to flat 1 row image
     // reshape(a,b) a = # of channel, 1 = num of rows
     // ex) [10 20 30 ] values indicating grayscale becomes [10 20 30 40 50 60 70 80 90]
     //     [40 50 60]
     //     [70 80 90]
     // because our Mat patches store each patch as 1 row.
     patch_mat = patch_mat.reshape(1,1);
     
     // if the window covers only white pixels, not black(target)
     // it adds to the patch. So the patches are inside the I - omega
     // Patches are NOT inside the ometa(massk, or target).
     if( countNonZero(mask_patch) == win_size * win_size )
     {
     for(int j =0; j < f_vec_len; j++)
     patches.at<uchar>(num_patches,j) = patch_mat.at<uchar>(0,j);
     num_patches++;
     
     }
     }
     }
     cout<<num_patches<<" patches created!"<<endl;
     */
    /*
     //compute C(p) i.e. by mean filtering
     Mat mask_gray_f, C_p, filled_f, filled;
     mask_gray.convertTo(mask_gray_f, CV_32FC1, 1./255.);
     filled = mask_gray.clone();
     
     
     int iter = 0;
     while(countNonZero(filled) != mask.rows * mask.cols)
     {
     
     
     filled.convertTo(filled_f, CV_32FC1, 1./255.);
     blur(filled_f, C_p, Size(win_size, win_size), Point(-1,-1) );
     
     // find boundaries!
     vector<vector<Point> > contours;
     Mat not_filled = 255 - filled;
     findContours(not_filled, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
     
     // find pixel with the highest priority
     float max_p_p = -1;
     int max_c_idx = -1;
     int max_p_idx = -1;
     
     
     for(int c = 0; c < contours.size(); c++)
     {
     for(int p = 0; p < contours[c].size(); p++)
     {
     Point this_p = contours[c][p];
     int last_point_idx = (p-1) < 0 ? contours[c].size() - 1: p-1;
     int next_point_idx = (p+1)%contours[c].size();
     Point last_p = contours[c][last_point_idx];
     Point next_p = contours[c][next_point_idx];
     Point boundary_dir_p = next_p - last_p;
     Point2f grad_p(grad_x.at<float>(this_p), grad_y.at<float>(this_p) );
     float this_c_p = C_p.at<float>(this_p);
     float priority = this_c_p * fabs(grad_p.x*boundary_dir_p.x + grad_p.y*boundary_dir_p.y);
     if(priority > max_p_p)
     {
     max_p_p = priority;
     max_c_idx = c;
     max_p_idx = p;
     }
     
     }
     }
     
     */
    
    return 0;
}


void showMatBasics(String srcName, String maskName) {
    // https://stackoverflow.com/questions/13484475/what-are-the-upper-and-lower-limits-and-types-of-pixel-values-in-opencv
    
    Mat src, src_gray, mask, mask_gray, grad_x, grad_y, inpaint_me;
    
    src = imread(srcName);
    mask = imread(maskName);
    src.copyTo(inpaint_me, mask);
    cvtColor(inpaint_me, inpaint_me, CV_BGR2Lab);//inpaint_me is now in Lab space
    cvtColor(src, src_gray, CV_BGR2GRAY);
    cvtColor(mask, mask_gray, CV_BGR2GRAY);
    Mat src_gray_f; // source gray floating-point
    src_gray.convertTo(src_gray_f, CV_32FC1, 1./255.);
    
    // Mat test = 30(row) x 30(col) initialized all the values to 25
    // Notice how <double> is used because CV_64FC1 64 Floating point channel 1 uses 64F, which is double
    cv::Mat test(cv::Size(30, 30), CV_64FC1);
    test = 25;
    cout << "\ntest at (0,0): " << test.at<double>(0, 0) << endl;
    cout << "test at (29,29): " << test.at<double>(test.rows - 1, test.cols - 1) << endl;
    cout << "test at (30,30) (Out of Index!): " << test.at<double>(test.rows, test.cols) << endl; // out of index
    
    imshow("src_gray", src_gray);
    cout << "print source_gray" << endl;
    cout << "src_gray at (0,0): " << (int)src_gray.at<uchar>(0,0) << endl;
    cout << "src_gray at (20,20): " << (int)src_gray.at<uchar>(20,20) << endl;
    cout << "src_gray at (225,339):" << (int)src_gray.at<uchar>(src_gray.rows - 1, src_gray.cols - 1) << endl;
    
    imshow("src_gray_f", src_gray_f);
    cout << "print src_gray_f" << endl;
    cout << "src_gray_f at (0,0): " << src_gray_f.at<float>(0,0) << endl;
    cout << "src_gray_f at (20,20): " << src_gray_f.at<float>(20,20) << endl;
    cout << "src_gray_f at (225,339):" << src_gray_f.at<float>(src_gray_f.rows - 1, src_gray_f.cols - 1) << endl;
    
    imshow("mask", mask);
    cout << "print mask" << endl;
    cout << "mask at (0,0): " << (int)mask.at<uchar>(0,0) << endl;
    cout << "mask at (20,20): " << (int)mask.at<uchar>(20,20) << endl;
    cout << "mask at (225,339):" << (int)mask.at<uchar>(mask.rows - 1, mask.cols - 1) << endl;
    cv::waitKey(0);
    
}
