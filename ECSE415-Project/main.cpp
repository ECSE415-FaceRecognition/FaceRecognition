#include <algorithm>
#include <fstream>
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
void seven_fold_cv(std::vector<std::string> &people, std::vector<std::vector<cv::string> > &folds);
void qmul_all_images_of_person(std::string person);

double do_lbp_chisq_match( std::vector<std::vector<LBPData> > folds, int level, std::vector<std::string> const& people_tmp);
double do_lbp_prob_match( std::vector<std::vector<LBPData> > histograms, LBPData test);

/* logging */
std::ofstream lbp_face_log;

// data used for probability comparisons
struct ProbData {
    cv::Mat mean;
    cv::Mat covar;
    std::string name;
};

int main()
{
	//load image and set directory - just for testing
	tinydir_dir dir;
	tinydir_open(&dir, QMUL_DIR);

	std::vector<std::string> people = getQmulNames();

	do_lbp_face_recognition(people);

    return 0;
}

void do_lbp_face_recognition(std::vector<std::string> const& people_tmp) {

	/* N people by X images per person */
	std::vector<std::vector<LBPData> > histograms;

	std::vector<std::vector<cv::string> > image_names = open_all_qmul_by_person(people_tmp);
	std::vector<std::vector<LBPData> > folds;

    /* open a file for logging */
	lbp_face_log.open("lbp_face_log.txt");


    /* shrink dataset */
    image_names.resize(5);


    for (auto &image : image_names) {
        image.resize(6);
    }

	/* get lbp histrograms of all images.
     * when needed, simply pass a copy to the data 
     */
	lbp_train(image_names, histograms, MAX_LEVELS);

    std::vector<LBPData> testing_images;
    for (auto &person : histograms) {
        testing_images.push_back(person[5]);
        person.resize(5);
    }


    do_lbp_prob_match(histograms, testing_images[0]);

    return;

	/* split into 7 training sets, preserving the order of people */
	folds.resize(NUM_FOLDS);
	for (int i = 0; i < histograms.size(); i++) {
		srand(0xDEADBEEF);	// make test deterministic
		seven_fold_cv(histograms[i], folds);
	}
	
    /* calculate LBP over levels up to MAX_LEVELS */
	for (int level = 1; level <= MAX_LEVELS; level++) {
        double average_rate = do_lbp_chisq_match(folds, level, people_tmp);
		std::cout << "Overall rate was " << average_rate / NUM_FOLDS << " for level " << level << std::endl;
	}
	system("pause");
}

/* open all qmul images of a particular person, as specified in the assignment
 * description
 */
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

double do_lbp_prob_match( std::vector<std::vector<LBPData> > histograms, LBPData test ) {

    int sz = histograms[0].size();
    std::vector<ProbData> gaussians;
    std::vector<cv::Mat> all_histogram_of_person;
    /* fill `gaussians with the covar and mean of all images of one person */
    for (auto &person : histograms) {
        all_histogram_of_person.clear();
        /* iterate through all images of a person */
        for (auto &image : person) {
            // std::cout << image.name << std::endl;
            cv::Mat tmp;
            for (auto &level_hist : image.hist) {
                tmp.push_back(level_hist);
            }
            all_histogram_of_person.push_back(tmp);
        }

        /* fill ProbData Struct */
        ProbData tmp;
        tmp.name = person[0].name;
        cv::calcCovarMatrix(all_histogram_of_person, tmp.covar, tmp.mean, CV_COVAR_NORMAL, 5);

        /* store covar and mean of this person */
        gaussians.push_back(tmp);
    }

    std::vector<cv::Mat> test_vector;
    ProbData tmp;

    cv::Mat person;
    tmp.name = test.name;
    for (auto &level : test.hist) {
        person.push_back(level);
    }

    test_vector.push_back(person);
    cv::calcCovarMatrix(test_vector, tmp.covar, tmp.mean, CV_COVAR_NORMAL, 5);

    std::cout << "size of covar from one histogram" << tmp.covar.size() << std::endl;
    std::cout << "size of mean from one histogram" << tmp.mean.size() << std::endl;

    gaussians.push_back(tmp);

    auto sz_covar = gaussians[0].covar.size();
    auto sz_mean = gaussians[0].mean.size();
    for (auto &gaussian : gaussians) {
        if (gaussian.mean.size() != sz_mean) {
            std::cout << gaussian.name << " has a strange mean size" << std::endl;
        }
        if (gaussian.covar.size() != sz_covar) {
            std::cout << gaussian.name << " has a strange covar size" << std::endl;
        }
    }
    

    // TODO diagonalize a matrix ?? http://cs229.stanford.edu/section/gaussians.pdf ??
    // TODO compute gaussian difference
    
    /*
     * max(p(I|s)) => probability of I (a person) given s (a multivariate gaussian of a testing
     * image)
     * 
     * => how to get p(s|I) using covar and mean? (
     * https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Non-degenerate_case)
     */

    for (int i=0; i < gaussians.size(); i++) {
        cv::Mat diff = (tmp.mean - gaussians[i].mean);
        cv::Mat exponent = -0.5 * diff.t() * gaussians[i].covar.inv(cv::DECOMP_SVD) * diff;

        cv::Mat result;
        exponent = cv::abs(exponent);
        cv::exp(exponent, result);


        std::cout << exponent << " for " << gaussians[i].name  <<  std::endl;
        std::cout << result << " for " << gaussians[i].name  <<  std::endl;
    }

    std::cout << "actual: " << tmp.name << std::endl;

    /*
     * => use same method to get p(s) from testing image
     * p(I|s) = p(s|I) / p(s)
     * p(I|s) = p(s|I) / sum(...)
     */

}

double do_lbp_chisq_match( std::vector<std::vector<LBPData> > folds, int level, std::vector<std::string> const& people_tmp) {
    /* set of folds to use for training */
	std::vector<std::vector<LBPData> > training_images;
    /* set of folds to use for testing */
	std::vector<LBPData> testing_images;
    /* to return */
	double average_rate = 0;
    /* variable to keep track of fold */
	int which_fold;
	/* create a set of training images from 6 of the 7 folds */
	for (int x = 0; x < NUM_FOLDS; x++) {
		training_images.clear();
		for (int fold=0; fold < (NUM_FOLDS-1); fold++) {
			which_fold = (fold + x) % NUM_FOLDS;
			training_images.push_back(folds[which_fold]);
		}
		which_fold = (NUM_FOLDS-1 + x) % NUM_FOLDS;
		testing_images = folds[which_fold];

		int guessed_correct = 0;

		/* run recognition for testing subsample */
		for (int i=0; i < testing_images.size(); i++) {
            /* get name and historam laters */
			std::vector<cv::Mat> test_person = testing_images[i].hist;  //  TODO set as reference
			std::string test_name = testing_images[i].name;
			std::string guessed = lbp_test(test_person, people_tmp, training_images, level);
		
			/* determine if lbp guessed properly. test_name is a file name, so we simply
				search in the file name for the guessed person */
			if (test_name.find(guessed) != std::string::npos) {
				guessed_correct++;
				//std::cout << "guessed " << guessed_correct << "correct" << std::endl;
			}
		}

		/* log the recognition rate */
		double rate = (((double)guessed_correct) / ((double)folds[(NUM_FOLDS - 1 + x) % NUM_FOLDS].size()));
		lbp_face_log << "For level " << level << " iteration " << x << "guessed" << guessed_correct << " of " << folds[(NUM_FOLDS - 1 + x) % NUM_FOLDS].size() 
			<< "with a rate of " << rate << std::endl;
		average_rate += rate;
	}
}
