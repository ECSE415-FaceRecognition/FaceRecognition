#pragma once

#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp> 

#include <string>
#include <vector>

void lbp_train(std::vector<std::vector<std::string>> const& people, std::vector<std::vector<std::vector<cv::Mat>>> &histograms, int levels);
std::string lbp_test(std::string const& test_file, std::vector<std::string> const& people, std::vector<std::vector<std::vector<cv::Mat>>> &histograms, int levels);