#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "lbp.h"
#include "tinydir.h"

// allow numeric limits to work
#undef max

const char QMUL_DIR[] = "QMUL/";

std::string get_image(std::string person, int tilt, int angle) {
	std::stringstream s, tilt_ss, angle_ss;
	tilt_ss << std::setfill('0') << std::setw(3) << tilt;
	angle_ss << std::setfill('0') << std::setw(3) << angle;

	s << QMUL_DIR << person << "/" << person.substr(0, person.size() - 4) << "_"  << tilt_ss.str() << "_" << angle_ss.str() << ".ras";
	return s.str();
}

void main()
{
	//load image and set directory - just for testing

	tinydir_dir dir;
	tinydir_open(&dir, QMUL_DIR);

	std::vector<std::string> people;

	// populate peopls with everyone from the QMUL dataset
	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (file.is_dir)
		{
			if (file.name[0] != '.') {
				people.push_back(file.name);
			}
		}
		tinydir_next(&dir);
	}

	tinydir_close(&dir);

	std::vector<cv::Mat> histograms, tests;

	/* train */
	for (int i=0; i < people.size(); i++) {
		
		std::string name = get_image(people[i], 120, 90);
		// open image
		cv::Mat im = cv::imread(name);
		
		//convert to greyScale
		cv::cvtColor(im, im, CV_RGB2GRAY);
		
		//compute spatial pyramid histogram for number of level
		histograms.push_back(getSpatialPyramidHistogram(im, 1));
	}

	/* testing */
	int test_person = 8;

	std::string name = get_image(people[test_person], 110, 90);
	std::cout << "testing: " << name << std::endl;
	// open image
	cv::Mat im = cv::imread(name);
		
	//convert to greyScale
	cv::cvtColor(im, im, CV_RGB2GRAY);
		
	//compute spatial pyramid histogram for number of level
	tests.push_back(getSpatialPyramidHistogram(im, 1));

	
	/* person 1 */
	double best = std::numeric_limits<double>::max();
	int person = -1;
	for (int i=0; i<histograms.size(); i++) {
		double diff = cv::compareHist(histograms[i], tests[0], CV_COMP_CHISQR);

		std::cout << "Difference was " << diff << ", opposed to best: " << best << std::endl;

		if (diff < best) {
			best = diff;
			person = i;
		}

	}

	std::cout << "The Person was most likely: " << people[person] << std::endl;
	if (person != test_person) {
		std::cout << "This was wrong" << std::endl;
	}

	std::system("pause");	


}