// Project

//includes
#include "lbp.h"

#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp> 

//namespaces
using namespace cv;
using namespace std;

Mat getSpatialPyramidHistogram(const Mat input, int levels){
	Mat spatialHistogram;
	//compute histogram for each level
	for (int i = 1; i < levels+1; i++){
		//calculate histogram of current level
		Mat currentLevelHist = computeImageLBP(input, i);
		
		//normalize
		normalize(currentLevelHist, currentLevelHist, 1, NORM_L2);

		//concatenate to complete Spatial Pyramid Histogram
		spatialHistogram.push_back(currentLevelHist);
	}

	//return full spatial pyramid histogram
	return spatialHistogram;
}


Mat computeImageLBP(const Mat input, int patchNumber){
	Mat imageHistogram;
	vector <Mat> imageSplit(patchNumber*patchNumber);
	//find size of images
	int patchWidth = input.cols / patchNumber;
	int patchHeight = input.rows / patchNumber;

	//split image
	int k = 0;
	for (int y = 0; y < patchNumber; y++){
		for (int x = 0; x < patchNumber; x++){
			imageSplit[k] = input(Rect(x*patchWidth, y*patchHeight, patchWidth, patchHeight));
			k++;
		}
	}

	//compute histogram for each image and concate to image Histogram;

	for (int i = 0; i < patchNumber*patchNumber; i++){
		//calculate local histogram
		Mat currentHist = computeLBP(imageSplit[i]);

		//normalize
		normalize(currentHist, currentHist, 1, NORM_L2);

		//concatenate to complete Image Histogram
		imageHistogram.push_back(currentHist);
	}

	//return image Histogram
	return imageHistogram;

}


Mat computeLBP(const Mat input){
	//compute LBP for every pixel in input matrix
	Mat LBPimage(input.size(), CV_64F);

	for (int x = 1; x < input.cols-1; x++){
		for (int y = 1; y < input.rows-1; y++){
		//extract surrounding 3x3 matrix for every pixel
			Mat pixelNeighbors = input(Rect(x - 1, y - 1, 3, 3));
			LBPimage.at<double>(y, x) = computePixelLBP(pixelNeighbors);
		}
	}
	
	//show image
	LBPimage.convertTo(LBPimage, CV_8U);
	//imshow("LBP image", LBPimage);
	//waitKey(0);

	//compute histogram
	Mat histogram;
	int nbins = 59; // 59 bins
	int histSize[] = { nbins }; //one dimension
	float range[] = {0, 255}; //up to 255 value
	const float *ranges[] = { range };
	int channels[] = { 0 };
	calcHist(&LBPimage, 1, channels, Mat(), histogram, 1, histSize, ranges);

	//return histogram
	return histogram;
}


double computePixelLBP(const Mat input){
		//compute LBP value for pixel
		Mat pixel;

		input.convertTo(pixel, CV_32S);
		//only for testing
		//Mat pixel = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);	
		//testing end

		int center = pixel.at<int>(1, 1);
		vector<int> LBPvec;

		//check every pixel and create 8bit array, starting at left top corner of matrix
		LBPvec.push_back(!(center < pixel.at<int>(0, 0)));
		LBPvec.push_back(!(center < pixel.at<int>(0, 1)));
		LBPvec.push_back(!(center < pixel.at<int>(0, 2)));
		LBPvec.push_back(!(center < pixel.at<int>(1, 2)));
		LBPvec.push_back(!(center < pixel.at<int>(2, 2)));
		LBPvec.push_back(!(center < pixel.at<int>(2, 1)));
		LBPvec.push_back(!(center < pixel.at<int>(2, 0)));
		LBPvec.push_back(!(center < pixel.at<int>(1, 0)));
		
		//check if there are more than two 0-1 or 1-0 transitions
			int transitions = 0;
			//check for every element but last
			for (int i = 0; i < LBPvec.size()-1; i++){
				if (LBPvec[i + 1] - LBPvec[i] != 0)
					transitions = transitions + 1;
			}
			//check for first and last element
			if (LBPvec[0] - LBPvec[LBPvec.size()-1] != 0){
					transitions = transitions + 1;
			}
		//compute LVP value
			double LVPvalue = 0;
			//if transitions are 2 or less, compute the LBP value, otherwise LVPvalue remains 0
			if (transitions <= 2){
				for (int i = 0; i < LBPvec.size(); i++){
					if (LBPvec[i] == 1){
						LVPvalue = LVPvalue + pow(2, (double)i);
					}
				}
			}
			//cout << LVPvalue << "\n";
		//return LVP value
		return LVPvalue;

}

