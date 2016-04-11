#include <algorithm>  
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
void seven_fold_cv(std::vector<std::string> &people, std::vector<std::vector<cv::string>> &folds);
void qmul_all_images_of_person(std::string person);

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
	std::vector<std::vector<cv::Mat>> histograms;

	std::vector<std::vector<cv::string>> image_names =  open_all_qmul_by_person(people);
	std::vector<std::vector<cv::string>> folds;

	/* split into 7 training sets */
	folds.resize(7);
	for (int i=0; i < image_names.size(); i++) {
		srand(time(0));
		seven_fold_cv(image_names[i], folds);
	}

	std::vector<std::vector<cv::string>> training_images;
	training_images.resize(people.size());
	for (int fold=0; fold < 6; fold++) {
		int i=0;
		for (int person=0; person<people.size(); person++) {
			while ( i < folds[fold].size() && (folds[fold][i].substr(5, people[person].size()) == people[person].substr(0, people[person].size()))) {
				training_images[person].push_back(folds[fold][i]);
				i++;
			}
		}
	}
	for (int levels = 1; levels <= 10; levels++) {
		histograms.clear();
		int guessed_correct = 0;
		lbp_train(training_images, histograms, levels);

		for (int i=0; i < folds[6].size(); i++) {
			std::string test_person = folds[6][i];
			std::string guessed = lbp_test(test_person, people, histograms, levels);
			//std::cout << "Guessed: " << guessed << ", " << "Actual person was: " << test_person << std::endl;
			if (test_person.find(guessed) != std::string::npos) {
				guessed_correct++;
			}
		}
		std::cout << "for " << levels << " guessed with a rate of " << (((double) guessed_correct)/ ((double) folds[6].size())) << std::endl;
	}

	system("pause");
}

void qmul_all_images_of_person(std::string person) {
	std::vector<std::string> tmp;
	tmp.push_back(person);

	auto open_test = open_all_qmul_by_person(tmp);

	const int index = 0;
	for (int i=0; i < open_test[index].size(); i++) {
		cv::Mat img = cv::imread(open_test[index][i]);
		cv::imshow(open_test[index][i], img);
	}

	cv::waitKey();

}

void seven_fold_cv(std::vector<std::string> &people, std::vector<std::vector<cv::string>> &folds) {
	std::random_shuffle(people.begin(), people.end());
	int i = people.size();
	while(i > 0) {
		i--;
		folds[i%7].push_back(people[i]);
	}
}
