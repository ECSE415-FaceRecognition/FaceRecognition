#include "eigenfaces.h"

Mat get_mean_image(vector<Mat> faces);

void generate_ef(vector<Mat> faces) {

	int size_in = faces.size();
	int img_h, img_w;
	img_h = faces[0].size().height;
	img_w = faces[0].size().width;
	Mat D = Mat::zeros(img_h*img_w, size_in, CV_64FC1);
	Mat mean_face = get_mean_image(faces);
	vector<Mat> diff_faces;
	for (int i = 0; i < faces.size(); i++) {
		Mat face;
		faces[i].convertTo(face, CV_64FC1);
		diff_faces.push_back(face - mean_face);
	}

	int size = diff_faces.size();

	cout << "size in is " << size << endl;
	for (int i = 0; i < size_in; i++) {
		for (int j = 0; j < img_h; j++) {
			for (int k = 0; k < img_w; k++) {
				D.at<double>(j*img_w + k,i) = diff_faces[0].at<double>(j, k);
			}
		}
	}

	cout << "size of mean face " << mean_face.size() << endl;
	cout << diff_faces[0].at<double>(4,6) << endl;
	cout << D.at<double>(4 * img_w + 6, 0) << endl;
	cout << "size of D " << D.size() << endl;
	

	Mat DD_t;
	// D x D_transpose
	DD_t = D*D.t();
	cout << DD_t.size() << endl;
	
	//do i need this?
	Mat covar, mean;
	calcCovarMatrix(DD_t, covar, mean, CV_COVAR_COLS);
	cout << covar.size() << endl;



	Mat eigen_vec;
	Mat eigen_val;

	if (eigen(DD_t, true, eigen_vec, eigen_val)) {
		cout << eigen_vec.size() << endl;
		cout << eigen_val.size() << endl;
	}
	else {
		cout << "error in eigen function" << endl;
		_exit(0);
	}
	
	//return DD_t;
}


Mat get_mean_image(vector<Mat> faces) {
	Mat conv;
	Mat result = Mat::zeros(faces[0].size().height, faces[0].size().width, CV_64FC1);

	for (int i = 0; i < faces.size(); i++) {
		faces[i].convertTo(conv, CV_64FC1);
		result += conv;
	}
	result *= (1.0 / faces.size());
	cout << "size of result = " << result.size() << endl;
	return result;
}