#include "eigenfaces.h"

void generate_ef(Mat D, Mat &eigen_vec, Mat &eigen_val);
Mat generate_flat_diff(vector<Mat> faces);
void sort_mat(const Mat &input, Mat &sorted, const Mat &indices);

Mat mean_face;
Mat coeffs;

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

	Mat eigen_vec_tmp;
	//Mat eigen_val;

	if (eigen(DD_t, true,eigen_val,eigen_vec_tmp)) {
		cout << eigen_vec_tmp.size() << endl;
		cout << eigen_val.size() << endl;
	}
	else {
		cout << "error in eigen function" << endl;
		exit(0);
	}
	
	eigen_vec = D*eigen_vec_tmp;
    
	cout << "eigen_vec_tmp = " << eigen_vec_tmp.size() << endl;

	//Mat source = Mat::eye(eigen_vec.size(), CV_64FC1);
//	Mat dst;
//    cout<<"Eigen_Vec is"<<eigen_val<<endl;
//	cv::sortIdx(eigen_val, dst, CV_SORT_DESCENDING + CV_SORT_EVERY_COLUMN);
//	cout << dst.size() << endl;
//	cv::sort(eigen_val, eigen_val, CV_SORT_DESCENDING + CV_SORT_EVERY_COLUMN);
//	sort_mat(eigen_vec_tmp, eigen_vec, dst);
//    cout<<"Sorted Eigen_vec is"<<eigen_val<<endl;
//	cout << "size of eigen_vec = " << eigen_vec.size() << endl;
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
	int img_h, img_w;
	img_h = faces[0].size().height;
	img_w = faces[0].size().width;
	Mat D = Mat::zeros(img_h*img_w, size_in, CV_64FC1);
	mean_face = get_mean_image(faces);
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

	coeffs = Mat::zeros(eigen_faces.size(), CV_64F);
	//Mat temp = ((int)faces.size(), (int)faces.size(), CV_64FC1, D);
	//Mat temp_ev = eigen_faces.reshape(1, 1);
	//cout << temp_ev.size() << endl;
    
    //Now project each training image onto eigen_faces
	for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < faces.size(); j++) {
			cout << "D.col(j) size = " << D.col(j).size() << endl;
			cout << "eigen_faces.col(i) size = " << eigen_faces.col(i).size() << endl;
			coeffs.at<double>(i, j) = D.col(j).dot(eigen_faces.col(i));
		}
	}
	return coeffs;
}

Mat get_mean_image(vector<Mat> faces) {
	Mat conv;
	Mat result = Mat::zeros(faces[0].size().height, faces[0].size().width, CV_64FC1);

	for (int i = 0; i < faces.size(); i++) {
		faces[i].convertTo(conv, CV_64FC1);
		result += conv;
	}
	result = result / faces.size();
	return result;
}

Mat test(Mat candidate)
{
//    vector<Mat> test_vec;
//    test_vec.push_back(candidate);
    
    Mat flat_candidate = Mat::zeros(1, candidate.rows*candidate.cols,CV_64F);
    candidate.reshape(0,1); //Flatten the input candidate image
    Mat flat_mean = mean_face.reshape(0,1);
    namedWindow( "Display window", WINDOW_AUTOSIZE );   // Create a window for display.
    imshow( "Display window", flat_candidate);
    cv::waitKey();
    flat_candidate = flat_candidate-flat_mean;
    
    //Now Project
    return candidate;
}