#pragma once

#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp> 

#include <string>
#include <vector>

struct LBPData {
	std::vector<cv::Mat> hist;
	std::string name;
};

struct LBPProbData {
    cv::Mat covar;
    cv::Mat mean;
    std::string name;
};

void lbp_train(std::vector<std::vector<std::string> > const& people, std::vector<std::vector<LBPData> > &histograms, int levels);
std::string lbp_test(std::vector<cv::Mat> test_person, std::vector<std::string> const& people, std::vector<std::vector<LBPData> > &histograms, int levels);
std::string lbp_prob_test(LBPProbData test_person, std::vector<std::string> const& people, std::vector<LBPProbData> &gaussians, int levels);
cv::Mat displayLBP(cv::Mat input);