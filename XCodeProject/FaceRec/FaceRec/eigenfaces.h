#pragma once
#ifndef _EIGENFACE__DEF_
#define _EIGENFACE__DEF_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//Mat mean_face;
//Mat coeffs;

Mat train(vector<Mat> faces);
Mat test(Mat candidate);
Mat get_mean_image(vector<Mat> faces);

#endif
