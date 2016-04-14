#include "eigenfaces.h"

void get_mean_image(vector<Mat> faces);
void generate_ef(Mat &D, Mat &eigen_vec, Mat &eigen_val);
Mat generate_flat_diff(vector<Mat> faces);
//Mat train(vector<Mat> faces);
//Mat test(Mat candidate);
void sort_mat(const Mat &input, Mat &sorted, const Mat &indices);

Mat mean_face;
Mat coefs;
//Mat eigen_faces;


void generate_ef(Mat &D, Mat &eigen_vec, Mat &eigen_val) {

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

	Mat eigen_vec_tmp;
	//Mat eigen_val;

	

	if (eigen(DD_t, true,eigen_val,eigen_vec_tmp)) {
		cout << eigen_vec_tmp.size() << endl;
		cout << eigen_val.size() << endl;
	}
	else {
		cout << "error in eigen function" << endl;
		_exit(0);
	}
	
	eigen_vec = D*eigen_vec_tmp;
	cout << "eigen_vec = " << eigen_vec.size() << endl;

	//Mat source = Mat::eye(eigen_vec.size(), CV_64FC1);
	/*Mat dst;
	cv::sortIdx(eigen_val, dst, CV_SORT_DESCENDING + CV_SORT_EVERY_COLUMN);
	cout << dst.size() << endl;
	cv::sort(eigen_val, eigen_val, CV_SORT_DESCENDING + CV_SORT_EVERY_COLUMN);
	sort_mat(eigen_vec_tmp, eigen_vec, dst);
	cout << "size of eigen_vec = " << eigen_vec.size() << endl;*/
	//return DD_t;
}

void sort_mat(const Mat &input, Mat &sorted, const Mat &indices) {
	cout << indices.rows << endl;
	sorted = Mat::zeros(input.size(), CV_64FC1);
	for (int i = 0; i < indices.rows; i++) {
		input.col(indices.at<int>(0, i)).copyTo(sorted.col(i));
	}
}

Mat generate_flat_diff(vector<Mat> faces) {
	int size_in = faces.size();
	cout << "number of faces = " << size_in << endl;
	int img_h, img_w;
	img_h = faces[0].size().height;
	img_w = faces[0].size().width;
	Mat D = Mat::zeros(img_h*img_w, size_in, CV_64FC1);
	get_mean_image(faces);

	//mean_face.convertTo(mean_face, CV_8U);
	/*imshow("test mean face", mean_face);
	waitKey(1);*/
	//mean_face.convertTo(mean_face, CV_64FC1);

	vector<Mat> diff_faces;
	for (int i = 0; i < faces.size(); i++) {
		Mat face;
		Mat diff;
		faces[i].convertTo(face, CV_64FC1);
		diff = face - mean_face;
		diff_faces.push_back(diff);
	}

	int size = diff_faces.size();

	cout << "size in is " << size << endl;
	for (int i = 0; i < size; i++) {
		diff_faces[i] = diff_faces[i].reshape(1, 1).t();
		cout << "diff_faces[i] size = " << diff_faces[i].size() << endl;
		diff_faces[i].copyTo(D.col(i));
		/*for (int j = 0; j < img_h; j++) {
			for (int k = 0; k < img_w; k++) {
				D.at<double>(j*img_w + k, i) = diff_faces[i].at<double>(j, k);
			}
		}*/
	}
	cout << "D size = " << D.size() << endl;
	return D;
}

Mat train(vector<Mat> faces) {
	//cols are resulting projection
	Mat eigen_val, eigen_faces;
	Mat D = generate_flat_diff(faces);
	generate_ef(D, eigen_faces, eigen_val);
	cout << "faces.size = " << faces.size() << endl;
	coefs = Mat::zeros(faces.size(), faces.size(), CV_64F);
	//Mat temp = ((int)faces.size(), (int)faces.size(), CV_64FC1, D);
	//Mat temp_ev = eigen_faces.reshape(1, 1);
	//cout << temp_ev.size() << endl;
	for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < faces.size(); j++) {
			//cout << "D.col(j) size = " << D.col(j).size() << endl;
			//cout << "eigen_faces.col(i) size = " << eigen_faces.col(i).size() << endl;
			coefs.at<double>(i, j) = D.col(j).dot(eigen_faces.col(i));
		}
	}
	//coefs is indexed eigen_face,face.


	//Mat face;
	//eigen_faces.col(0).copyTo(face);
	//face = face.reshape(1, 100);
	//face.convertTo(face, CV_64FC1);

	//imshow("face0: eigenspace", face);
	//aitKey(1);

	return eigen_faces;
}

void get_mean_image(vector<Mat> faces) {
	
	mean_face = Mat::zeros(faces[0].size().height, faces[0].size().width, CV_64FC1);
	for (int i = 0; i < faces.size(); i++) {
		Mat conv;
		faces[i].convertTo(conv, CV_64FC1);
		mean_face += conv;
	}
	mean_face *= (1.0 / faces.size());
	cout << "size of result = " << mean_face.size() << endl;
	//return result;
}

int test(Mat &candidate, Mat &eigen_faces){

	Mat flat_candidate;
	//namedWindow("candidate", WINDOW_AUTOSIZE);   // Create a window for display.
	//imshow("candidate", candidate);
	//cv::waitKey(1);

	candidate.convertTo(candidate, CV_64FC1);
	flat_candidate = candidate - mean_face;
	flat_candidate = flat_candidate.reshape(1, 1).t(); //Flatten the input candidate image
	//Mat flat_mean = mean_face.reshape(1, 1).t();
	
	cout << "fc size = " << flat_candidate.size() << endl;
	
	//flat_candidate = flat_candidate.reshape(1, 100);
	//namedWindow("Display diff face", WINDOW_AUTOSIZE);   // Create a window for display.
	//imshow("Display diff face", flat_candidate);
	//cv::waitKey(1);

	//Now Project
	Mat test_coefs = Mat::zeros(coefs.rows, 1, CV_64F);
	cout << "eigen_faces type = " << eigen_faces.type() << endl;
	cout << "fc type = " << flat_candidate.type() << endl;
	for (int i = 0; i < eigen_faces.cols; i++) {
			//cout << "test.size = " << flat_candidate.size() << endl;
			//cout << "eigen_faces.col(i) size = " << eigen_faces.col(i).size() << endl;
			test_coefs.at<double>(i,1) = flat_candidate.dot(eigen_faces.col(i));
		}
	//test_coefs is indexed as face,eigen_face

	cout << "coefs size = " << coefs.size() << endl;
	cout << "test_coefs size = " << test_coefs.size() << endl;
	
	int min = INT_MAX;
	int min_id = -1;
	for (int i = 0; i < coefs.cols; i++) {
		if (abs(norm(test_coefs, coefs.col(i), NORM_L2)) < min) {
			min = abs(norm(test_coefs, coefs.col(i), NORM_L2));
			min_id = i;
		}
	}
	if (min_id == -1) {
		std::cout << "Error" << std::endl;
		
	}

	/*Mat result;
	eigen_faces.col(min_id).copyTo(result);
	Mat check = test_coefs.t()*eigen_faces.t();
	cout << "check size = " << check.size() << endl;
	check = check.t();
	check = check.reshape(1, 100);
	cout << "check size = " << check.size() << endl;
	check.convertTo(result,CV_8UC1);*/
	return min_id;
}