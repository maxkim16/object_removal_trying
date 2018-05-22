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
void showContoursPractice(String srcName, String maskName);
// https://stackoverflow.com/questions/6302171/convert-uchar-mat-to-float-mat-in-opencv
// https://stackoverflow.com/questions/13484475/what-are-the-upper-and-lower-limits-and-types-of-pixel-values-in-opencv
// https://stackoverflow.com/questions/7899108/opencv-get-pixel-channel-value-from-mat-image
// https://stackoverflow.com/questions/8449378/finding-contours-in-opencv
// https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html)
/*
 * Basic stuff about Mat
 * Mat (Row, Col)
 */

RNG rng(12345);
int main(int argc, char* argv[])
{
    //showMatBasics(argv[1], argv[2]);
    
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
    Mat src_gray_f; // source gray 32 bit floating-point 1 channel
    src_gray.convertTo(src_gray_f, CV_32FC1, 1./255.);
    
    // grad_x is a derivative of src_gray.
    // it shows sudden changes in x-direction
    // grad_y shows sundden changes in y-direction
    // http://mccormickml.com/2013/02/26/image-derivative/
    // when code it RESCALE!!! bc 0 - 255 = -255
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
    
    
    //COMPUTE C(p) i.e. by mean filtering
    Mat mask_gray_f, C_p, filled_f, filled;
    // CV_32FC1 => float, not double, value [0.0 - 1.0] channel 1
    mask_gray.convertTo(mask_gray_f, CV_32FC1, 1./255.);
    //cout << "print mask_gray_f" << endl;
    //cout << "mask_gray_f at (0,0): " << mask_gray_f.at<float>(0,0) << endl;
    //cout << "mask_gray_f at (20,20): " << mask_gray_f.at<float>(20,20) << endl;
    //cout << "mask_gray_f at (150,250) black space: " << mask_gray_f.at<float>(150,250) << endl;
    
    // filled is mask_gray float values
    filled = mask_gray.clone();
    
    
    Mat not_filled = 255 - filled;
    /*
    imshow("filled", filled);
    cout << "filled" << endl;
    cout << "filled at (0,0) black space: " << (int)filled.at<uchar>(0,0) << endl;
    cout << "filled at (150,250) black space: " << (int)filled.at<uchar>(150,250) << endl;
    cout << "filled at (150,300) black space: " << (int)filled.at<uchar>(150,300) << endl;
    cout << "filled at (170,300) black space: " << (int)filled.at<uchar>(170,300) << endl;
    cout << "filled at (33,133) black space: " << (int)filled.at<uchar>(150,251) << endl;
    
    imshow("not filled", not_filled);
    cout << "not filled" << endl;
    cout << "not filled at (0,0) black space: " << (int)not_filled.at<uchar>(0,0) << endl;
    cout << "not filled at (150,250) black space: " << (int)not_filled.at<uchar>(150,250) << endl;
    cout << "not filled at (150,300) black space: " << (int)not_filled.at<uchar>(150,300) << endl;
    cout << "not filled at (170,300) black space: " << (int)not_filled.at<uchar>(170,300) << endl;
    cout << "not filled at (33,133) black space: " << (int)not_filled.at<uchar>(150,251) << endl;
    cv::waitKey(0);
     */
    
    int iter = 0;
    //showContoursPractice(argv[1], argv[2]);
    // cout<<"here33"<<endl;
    // this while loop has 2 for loops
    // first for loop finds the highest priority patch on the fill front
    // second for loop finds the min distance and inpaint
    // loops untill all the target regions are filled.
    // iterate till the target region (only zeros) are all filled with confidence values
    // so Fill the target region with values!!!
    while(countNonZero(filled) != mask.rows * mask.cols)
    {
        // convert filled to CV_32FC1(rescale 1./255.) and then store it in filled_f
        filled.convertTo(filled_f, CV_32FC1, 1./255.);
        // apply mean filter(box filter)
        // filled_f now has 1's in the I-omega, floating point values around the boundary, and 0's in the middle of the target
        // that is far from the boundary
        blur(filled_f, C_p, Size(win_size, win_size), Point(-1,-1) );
        
     
        // find boundaries!
        vector<vector<Point> > contours;
        // not_filled = black regions become white and white regions becoem black
        Mat not_filled = 255 - filled;
        // get contours in the white region. Each contour is stored as a vector of points.
        findContours(not_filled, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        
        // FIND PIXEL WITH THE HEIGHEST PRIORITY
        // P(p) = C(p) * D(p)
        // C(p) = confidence value, how far the pixel is from the boundary
        // D(p) indicates the continuation of edge, that is, high frequency( sudden change of intensities)
        float max_p_p = -1;  // P(p)
        int max_c_idx = -1;  // row
        int max_p_idx = -1;  // col
        

        // FIND THE POINT WITH THE HIGHEST PRIORITY AND SAVE IT IN max_c_idx, max_p_idx, max
        // c = the number of contours.
        for(int c = 0; c < contours.size(); c++)
        {
            // the number of points in the current contour, which is c
            for(int p = 0; p < contours[c].size(); p++)
            {
                // going through every point in the current contour
                Point this_p = contours[c][p]; // current point
                // conditional operator => condition ? result_if_true : result_if_false
                // lasts_point_idx = current point - 1
                // it starts from the last point 178, then it becomes 0, 1, 2, ,,, 178
                int last_point_idx = (p-1) < 0 ? contours[c].size() - 1: p-1;
                // ex) size = 178 => 1,2,3, .... when it's 178, it becomes 0 because of mod
                int next_point_idx = (p+1) % contours[c].size();
                // previous point
                Point last_p = contours[c][last_point_idx];
                // next point
                Point next_p = contours[c][next_point_idx];
                //[241, 190] - [242, 188] = [1, -2] = next_boundary_dir_p
                // [1, -2] means x is going up and y is going down
                Point boundary_dir_p = next_p - last_p;
                // detect sudden changes (edges), it detects high frequencies
                Point2f grad_p(grad_x.at<float>(this_p), grad_y.at<float>(this_p) );
                
                float this_c_p = C_p.at<float>(this_p);
                // calculate C(p) * D(p) = Close to I-target * suddden changes(edges)
                // fabs returns the absolute value
                // (grad_p.x*boundary_dir_p.x + grad_p.y*boundary_dir_p.y) returns sudden change(direction + intensity) in x and y direction combined
                // direction is based on boundary_dir_p. and intensity change is depends on grad_x. and grad.y
                // so C(p) * D(p)
                float priority = this_c_p * fabs(grad_p.x*boundary_dir_p.x + grad_p.y*boundary_dir_p.y);
                if(priority > max_p_p)
                {
                    max_p_p = priority;  // get the highest priority
                    max_c_idx = c;  // get the address of the point with the highest priority
                    max_p_idx = p;  // get the address of the point with the highest priority
                }
            
            }
        }
        
        // (find min idst)
        // Given the point with the highest priority on the fill front, find the closest patch in the source region
        // that is most similar to the target
        // find min distance
        
        // start point is in the middle of the patch
        Point start = contours[max_c_idx][max_p_idx] - Point(win_size/2, win_size/2);
        // contour window patch
        Mat max_mask_patch = filled(Range(start.y, start.y+win_size), Range(start.x, start.x+win_size) ).clone();
        // contour window patch
        Mat max_patch_mat  = inpaint_me(Range(start.y, start.y+win_size),Range(start.x, start.x+win_size) ).clone();
        
        // [1 2 3 ]
        // [4 5 6 ]
        // [7 8 9 ] becomes [1 2 3 4 5 6 7 8 9] after reshape(1,1)
        // (a,b) . a = # num of colors, b = nth element
        max_mask_patch = max_mask_patch.reshape(1,1); // filled flat
        max_patch_mat =  max_patch_mat.reshape(1,1); // inpaint_me flat
        
        int min_dist = 255*255*3 + 1;
        int min_dist_idx = 0;
        // num_patches = patches in I - target region
        // y = nth pathch
        for(int y = 0; y < num_patches; y++)
        {
            int dist = 0;
            // f_vec_len = windowsSize(nm) * numOfChannels
            // x = xth point in nth patch
            for(int x = 0; x < f_vec_len; x++) // x up to 243 (9 * 9 * 3)
            {
                // x/3 = 0, 0, 0, 1, 1, 1, 2, 2, 2, ,,,
                // x/3 > 0 means if it's filled already, which indicates grayscale 255
                // if the point if filled already
                if(max_mask_patch.at<uchar>(0,x/3) > 0 )
                {
                    // get the nth patch and nth x
                    // ex) [x1 x2 x3]
                    //     [x4 x5 x6]   =>  [x1 x2 x3 x4 x5 x6]
                    
                    // here we are trying to find the best patch in I - target region that is most similar
                    // to the neighborhood of the pixel on the contour.
                    // diff means the difference between the point in yth patch and the current neighborhood
                    // dist is the total sum of all the differences. So if diff is really small
                    // that means the yth patch is very similar to the current neighborhood of the current pixel
                    // on the contour/
                    // so min_dist_idx becomes y.
                    // that's the patch we want to copy.
                    int diff = patches.at<uchar>(y,x) - max_patch_mat.at<uchar>(0,x);
                    dist += diff+diff;
                }
            }
            // if the point is not filled
            if(dist < min_dist)
            {
                min_dist = dist;
                // y indicates the nth patch. So it just finds the closest path near the point with the
                // highest priority
                min_dist_idx = y;
            }
        }
        
        
        // the point on the fill front with the highest priority
        Point fill_me = contours[max_c_idx][max_p_idx];
        // once filled becomes 255(white) it means that point in the region is inpainted!
        // we do it until all the region in the target is inpainted!
        filled.at<uchar>(fill_me) = 255;
        // inpaint_me is the mask, but just in LAB
        // L – Lightness ( Intensity ).
        // a – color component ranging from Green to Magenta.
        // b – color component ranging from Blue to Yellow.
        // !!! Only one point is copied, not the whole patch
        inpaint_me.at<Vec3b>(fill_me) = patches.at<Vec3b>(min_dist_idx, (win_size*win_size/2)); // win*win/2 means the middle pixel in the center
        //Vec3b bgrPixel2 = mask.at<Vec3b>(150, 250);
        //cout << "mask at (0,0) B: " << (uchar)bgrPixel[0] << " G: " << (uchar)bgrPixel[1] << " R: " << (uchar)bgrPixel[2] << endl;
        iter++;
        cout<<"Iter: "<<iter<<endl;
        
        
        
        
        //Save temporary results in save/ folder
        if(iter % 100 == 0)
        {
            Mat temp = inpaint_me.clone();
            cvtColor(temp, temp, CV_Lab2BGR);
            char name[256]={0};
            sprintf(name,"save/%d.png",iter);
            imwrite(name, temp);
            
        }

        cvtColor(inpaint_me, inpaint_me, CV_Lab2BGR);
        imwrite(argv[3], inpaint_me);
    }
    return 0;
}

// Detect the boundary, the edge of the target called "FILL FRONT"
// to assign patch priorities to the patches on the fill front ( the boundary)
void calculateC_p(String srcName, String maskName) {
    Mat src, src_gray, mask, mask_gray, grad_x, grad_y, inpaint_me;
    
    src = imread(srcName);
    mask = imread(maskName);
    src.copyTo(inpaint_me, mask);
    cvtColor(inpaint_me, inpaint_me, CV_BGR2Lab);//inpaint_me is now in Lab space
    cvtColor(src, src_gray, CV_BGR2GRAY);
    cvtColor(mask, mask_gray, CV_BGR2GRAY);
    Mat src_gray_f; // source gray 32 bits floating-point
    src_gray.convertTo(src_gray_f, CV_32FC1, 1./255.);
    
    //COMPUTE C(p) i.e. by mean filtering
    Mat mask_gray_f, C_p, filled_f, filled;
    // CV_32FC1 => float, not double, value [0.0 - 1.0] channel 1
    mask_gray.convertTo(mask_gray_f, CV_32FC1, 1./255.);
    cout << "print mask_gray_f" << endl;
    cout << "mask_gray_f at (0,0): " << mask_gray_f.at<float>(0,0) << endl;
    cout << "mask_gray_f at (20,20): " << mask_gray_f.at<float>(20,20) << endl;
    cout << "mask_gray_f at (150,250) black space: " << mask_gray_f.at<float>(150,250) << endl;
    
    
    // filled is mask_gray float values
    filled = mask_gray.clone();
    
    filled.convertTo(filled_f, CV_32FC1, 1./255.);
    blur(filled_f, C_p, Size(win_size, win_size), Point(-1,-1) );
    
    imshow("filled_f", filled_f);
    cout << "filled_f" << endl;
    cout << "filled_F at (0,0) black space: " << filled_f.at<float>(0,0) << endl;
    cout << "filled_F at (150,250) black space: " << filled_f.at<float>(150,250) << endl;
    cout << "filled_f at (150,300) black space: " << filled_f.at<float>(150,300) << endl;
    cout << "filled_f at (170,300) black space: " << filled_f.at<float>(170,300) << endl;
    cout << "filled_f at (33,133) black space: " << filled_f.at<float>(33,133) << endl;
    
    imshow("C_p", C_p);
    cout << "c_p" << endl;
    cout << "c_p at (0,0) black space: " << C_p.at<float>(0,0) << endl;
    cout << "c_p at (150,250) black space: " << C_p.at<float>(150,250) << endl;
    cout << "c_p at (150,300) black space: " << C_p.at<float>(150,300) << endl;
    cout << "c_p at (170,300) black space: " << C_p.at<float>(170,300) << endl;
    cout << "c_p at (200,260) black space: " << C_p.at<float>(200,260) << endl;
    cout << "c_p at (200,270) black space: " << C_p.at<float>(200,270) << endl;
    cout << "c_p at (200,290) black space: " << C_p.at<float>(200,290) << endl;
    cout << "c_p at (200,330) black space: " << C_p.at<float>(200,330) << endl;
    
    cout << "c_p at (33,133) black space: " << C_p.at<float>(33,133) << endl;
    
    Mat a = C_p(Range(0, 120), Range(0, 330)).clone();
    imshow("a", a);
    cv::waitKey(0);
    
    /*
     * Black pixels that are closed to white pixels (which are edges) will ahve strong values like 0.9, 0.8,,
     * Black pixels surrounded only by black pixels still remain black because their neighborhood is all black
     * so the mean value does not do anything.
     */
    
    for (int i = 120; i < 121; i++) {
        for (int j = 133; j < 220; j++) {
            cout << "(" << i << ")" << "(" << j << ")" << " = " << C_p.at<float>(i,j) << endl;
        }
    }

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
    Mat src_gray_f; // source gray 32 bits floating-point
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
    // show RGB values of (0,0) white and (150,250) black
    Vec3b bgrPixel = mask.at<Vec3b>(0, 0);
    Vec3b bgrPixel2 = mask.at<Vec3b>(150, 250);
    //cout << "mask at (0,0) B: " << (uchar)bgrPixel[0] << " G: " << (uchar)bgrPixel[1] << " R: " << (uchar)bgrPixel[2] << endl;
    cout << "mask at (0,0) B: " << (int)bgrPixel[0] << " G: " << (int)bgrPixel[1] << " R: " << (int)bgrPixel[2] << endl;
    cout << "mask at (150,250) B: " << (int)bgrPixel2[0] << " G: " << (int)bgrPixel2[1] << " R: " << (int)bgrPixel2
    [2] << endl;
    cout << "print mask" << endl;
    cout << "mask at (0,0): " << (int)mask.at<uchar>(0,0) << endl;
    cout << "mask at (20,20): " << (int)mask.at<uchar>(20,20) << endl;
    cout << "mask at (150,250) black space: " << (int)mask.at<uchar>(150,250) << endl;
    cout << "mask at (150,300) black space: " << (int)mask.at<uchar>(150,300) << endl;
    cout << "mask at (170,300) black space: " << (int)mask.at<uchar>(170,300) << endl;
    cout << "mask at (225,339):" << (int)mask.at<uchar>(mask.rows - 1, mask.cols - 1) << endl;
    
    imshow("mask_gray", mask_gray);
    cout << "print mask_gray" << endl;
    cout << "mask_gray at (0,0): " << (int)mask_gray.at<uchar>(0,0) << endl;
    cout << "mask_gray at (20,20): " << (int)mask_gray.at<uchar>(20,20) << endl;
    cout << "mask_gray at (150,250) black space: " << (int)mask_gray.at<uchar>(150,250) << endl;
    cout << "mask_gray at (150,300) black space: " << (int)mask_gray.at<uchar>(150,300) << endl;
    cout << "mask_gray at (170,280) black space: " << (int)mask_gray.at<uchar>(170,280) << endl;
    cout << "mask_gray at (225,339):" << (int)mask_gray.at<uchar>(mask_gray.rows - 1, mask_gray.cols - 1) << endl;
    
    imshow("inpaint", inpaint_me);
    cout << "print inpaint" << endl;
    cout << "inpaint at (0,0): " << (int)inpaint_me.at<uchar>(0,0) << endl;
    cout << "inpaint at (20,20): " << (int)inpaint_me.at<uchar>(20,20) << endl;
    cout << "inpaint at (150,250) black space: " << (int)inpaint_me.at<uchar>(150,250) << endl;
    cout << "inpaint at (225,339):" << (int)inpaint_me.at<uchar>(inpaint_me.rows - 1, inpaint_me.cols - 1) << endl;
    cv::waitKey(0);
    
}

void showContoursPractice(String srcName, String maskName) {
    Mat src, src_gray, mask, mask_gray, grad_x, grad_y, inpaint_me;
    
    src = imread(srcName);
    mask = imread(maskName);
    src.copyTo(inpaint_me, mask);
    cvtColor(inpaint_me, inpaint_me, CV_BGR2Lab);//inpaint_me is now in Lab space
    cvtColor(src, src_gray, CV_BGR2GRAY);
    cvtColor(mask, mask_gray, CV_BGR2GRAY);
    Mat src_gray_f; // source gray 32 bits floating-point
    src_gray.convertTo(src_gray_f, CV_32FC1, 1./255.);
    
    Mat mask_gray_f, C_p, filled_f, filled;
    mask_gray.convertTo(mask_gray_f, CV_32FC1, 1./255.);
    filled = mask_gray.clone();
    Mat not_filled = 255 - filled;
    
    filled.convertTo(filled_f, CV_32FC1, 1./255.);
    // apply mean filter(box filter)
    // filled_f now has 1's in the I-omega, floating point values around the boundary, and 0's in the middle of the target
    // that is far from the boundary
    blur(filled_f, C_p, Size(win_size, win_size), Point(-1,-1) );
    
    
    // find boundaries!
    vector<vector<Point> > contours;
    // not_filled = black regions become white and white regions becoem black
    // get contours in the white region. Each contour is stored as a vector of points.
    findContours(not_filled, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    
    // FIND PIXEL WITH THE HEIGHEST PRIORITY
    // P(p) = C(p) * D(p)
    // C(p) = confidence value, how far the pixel is from the boundary
    // D(p) indicates the continuation of edge, that is, high frequency( sudden change of intensities)
    float max_p_p = -1;  // P(p)
    int max_c_idx = -1;  // row
    int max_p_idx = -1;  // col
    
    vector<Vec4i> hierarchy;
    int thresh = 100;
    Mat canny_output;
    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }
    
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    cv::waitKey(0);
    
}
