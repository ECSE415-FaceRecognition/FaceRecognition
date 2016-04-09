#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "lbp.h"
#include "tinydir.h"
#include "util.h"

// allow numeric limits to work
#undef max

void do_lbp_face_recognition(std::vector<std::string> const& people);

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

	do_lbp_face_recognition(people);

}

void do_lbp_face_recognition(std::vector<std::string> const& people) {
	std::vector<cv::Mat> histograms;
	const std::string PERSON = "AdamBGrey";
	int levels = 1;

	lbp_train(people, histograms, levels);
	std::string test_person = get_image_qmul(PERSON, 60, 80);



	lbp_test(test_person, people, histograms, levels);
	std::cout << "Actual person was: " << PERSON << std::endl;


	system("pause");
}