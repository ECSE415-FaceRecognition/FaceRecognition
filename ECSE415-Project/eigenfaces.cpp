#include "eigenfaces.h"

Mat get_mean_image(vector<Mat> faces);
void generate_ef(Mat D, Mat &eigen_vec, Mat &eigen_val);
Mat generate_flat_diff(vector<Mat> faces);
Mat train(vector<Mat> faces);

void generate_ef(Mat D, Mat &eigen_vec, Mat &eigen_val) {

	//cout << D << endl;
	cout << "size of D " << D.size() << endl;
	cout << "D columns = " << D.cols << endl;
	cout << "D rows = " << D.rows << endl;
	

	Mat DD_t;
	//D x D_transpose
	DD_t = D.t()*D;
	//cout << DD_t.size() << endl;
	
	//do i need this?

	//DD_t.convertTo(DD_t, CV_32FC1);

	//Mat covar, mean;
	//calcCovarMatrix(DD_t, covar, mean, CV_COVAR_SCRAMBLED | CV_COVAR_COLS);
	
	//cout << covar.size() << endl;

	//covar.convertTo(covar, CV_32FC1);

	//Mat eigen_vec;
	//Mat eigen_val;

	

	if (eigen(DD_t, true, eigen_vec, eigen_val)) {
		cout << eigen_vec.size() << endl;
		cout << eigen_val.size() << endl;
	}
	else {
		cout << "error in eigen function" << endl;
		_exit(0);
	}
	
	//Mat source = Mat::eye(eigen_vec.size(), CV_64FC1);
	Mat dst;

	cv::sortIdx(eigen_vec, dst, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW);
	cout << dst.size() << endl;
	cv::sort(eigen_vec, eigen_vec, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW);
	//return DD_t;
}

Mat generate_flat_diff(vector<Mat> faces) {
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
				D.at<double>(j*img_w + k, i) = diff_faces[0].at<double>(j, k);
			}
		}
	}
	return D;
}

Mat train(vector<Mat> faces) {
	//cols are resulting projection
	Mat eigen_val, eigen_faces;
	Mat D = generate_flat_diff(faces);
	generate_ef(D, eigen_faces, eigen_val);

	Mat coefs = Mat::zeros(faces.size(),faces.size(), CV_64F);




	/*for (int i = 0; i < faces.size(); i++) {

	}*/
	return coefs;
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