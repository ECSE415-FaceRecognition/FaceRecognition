#pragma once
#ifndef _EIGENFACE__DEF_
#define _EIGENFACE__DEF_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat train(vector<Mat> faces);
int test(Mat &candidate, Mat &eigen_faces);

#endif
