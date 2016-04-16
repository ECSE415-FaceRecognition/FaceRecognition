#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void eigenface_std(vector<Mat> &faces, Mat &test) {

	vector<int> labels;

	for (int i = 0; i < 31; i++) {
		labels.push_back(i);
	}

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(faces, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(test);
	cout << "label predicted = " << predictedLabel << endl;
}






