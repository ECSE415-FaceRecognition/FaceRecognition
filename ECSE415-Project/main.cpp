#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "lbp.h"
#include "tinydir.h"

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
			people.push_back(file.name);
		}
		tinydir_next(&dir);
	}

	tinydir_close(&dir);

	// start at 2 to ignore "." and ".."
	for (int i=2; i < people.size(); i++) {
		
		std::string name = get_image(people[i], 120, 90);
		std::cout << "opening: " << name << std::endl;
		// open image
		cv::Mat im = cv::imread(name);
		
		//convert to greyScale
		cv::cvtColor(im, im, CV_RGB2GRAY);
		
		//compute spatial pyramid histogram for number of level
		cv::Mat spatialHist = getSpatialPyramidHistogram(im, 1);
	}

	cv::waitKey();
	std::system("pause");

	//open all files with pose _90_90

	//wait to exit
	

}