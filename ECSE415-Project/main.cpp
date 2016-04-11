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

const int NUM_FOLDS = 7;
const int MAX_LEVELS = 1;

void do_lbp_face_recognition(std::vector<std::string> const& people);
void seven_fold_cv(std::vector<std::string> &people, std::vector<std::vector<cv::string>> &folds);
void qmul_all_images_of_person(std::string person);

void main()
{
	//load image and set directory - just for testing
	tinydir_dir dir;
	tinydir_open(&dir, QMUL_DIR);

	std::vector<std::string> people = getQmulNames();

	do_lbp_face_recognition(people);

}

void do_lbp_face_recognition(std::vector<std::string> const& people_tmp) {

	/* N people by X images per person */
	std::vector<std::vector<LBPData>> histograms;

	std::vector<std::vector<cv::string>> image_names = open_all_qmul_by_person(people_tmp);
	std::vector<std::vector<LBPData>> folds;

	image_names.resize(10);
	for (auto &image : image_names) {
		image.resize(21);
	}

	/* get lbp histrograms of all images*/
	lbp_train(image_names, histograms, MAX_LEVELS);

	/* split into 7 training sets, preserving the order of people */
	folds.resize(NUM_FOLDS);
	for (int i = 0; i < histograms.size(); i++) {
		srand(time(0));
		seven_fold_cv(histograms[i], folds);
	}

		std::vector<std::vector<LBPData>> training_images;
		std::vector<LBPData> testing_images;

		/* create a set of training images from 6 of the 7 folds */
		int which_fold;
		for (int x = 0; x < NUM_FOLDS; x++) {
			training_images.clear();
			for (int fold=0; fold < (NUM_FOLDS-1); fold++) {
				which_fold = (fold + x) % NUM_FOLDS;
				training_images.push_back(folds[which_fold]);
			}
			which_fold = (NUM_FOLDS-1 + x) % 7;
			testing_images = folds[which_fold];

			/* run lbp recognition for a single set of folds */
			/* clear histograms from last session */
			//histograms.clear();
			int guessed_correct = 0;

			/* train on this set of training subsamples */
			

			/* run recognition for testing subsample */
			for (int i=0; i < testing_images.size(); i++) {
				std::string test_person = testing_images[i].name;
				std::string guessed = lbp_test(test_person, people_tmp, training_images, MAX_LEVELS);
			
				/* determine if lbp guessed properly. test_person is a file name, so we simply
					search in the file name for the guessed person */
				if (test_person.find(guessed) != std::string::npos) {
					guessed_correct++;
					//std::cout << "guessed " << guessed_correct << "correct" << std::endl;
				}
			}

			/* in order to log the recognition rate */
			std::cout << "for " << MAX_LEVELS << " guessed with a rate of " << (((double)guessed_correct) / ((double)folds[NUM_FOLDS-1].size())) << std::endl;
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
