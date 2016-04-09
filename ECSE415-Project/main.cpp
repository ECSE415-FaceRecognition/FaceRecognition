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
void test_util(std::vector<std::string> const& people);

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

	test_util(people);

	//do_lbp_face_recognition(people);

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

void test_util(std::vector<std::string> const& people) {
	auto open_test = open_all_qmul_by_pose(people);

	const int index = 0;
	for (int i=0; i < open_test[index].size(); i++) {
		cv::Mat img = cv::imread(open_test[index][i]);
		cv::imshow(open_test[index][i], img);
	}

	cv::waitKey();

}
