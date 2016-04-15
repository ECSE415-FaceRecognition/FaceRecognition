#include "eigenfaces.h"

Mat generate_flat_diff(vector<Mat> faces);
//Mat train(vector<Mat> faces);
//Mat test(Mat candidate);
void sort_mat(const Mat &input, Mat &sorted, const Mat &indices);

Mat mean_face;
Mat coeff;
Mat eigenfaces;
double num_eig = 20;

Mat train(vector<Mat> faces) {
    cout<<"Training"<<endl;
    Mat flat_images = Mat::zeros(faces.size(), faces[0].rows*faces[0].cols, CV_64F);
    
    //Flatten the images
    int col_count = 0;
    for(int i = 0; i<faces.size(); i++)
    {
        Mat temp = Mat::zeros(faces[0].rows, faces[0].cols, CV_64F);
        col_count = 0;
        faces[i].copyTo(temp);

        //Convert input images;
        temp.convertTo(temp, CV_64FC1);
        
        //Now reshape the face and copy to temp mat;
        temp.reshape(0,1);
        for(int j = 0; j<faces[i].rows*faces[i].cols; j++)
        {
            flat_images.at<double>(i, col_count) = temp.at<double>(0,col_count);
            col_count = col_count + 1;
        }
    }
    
    //  Compute the mean face
    mean_face = Mat::zeros(1, flat_images.cols, CV_64FC1);
    for(int i = 0; i<mean_face.cols; i++)
    {
        mean_face.at<double>(0, i) = mean(flat_images.col(i))[0];
    }
    
    //  Now compute diff_faces (face[i] - mean)
    Mat diff_faces = Mat::zeros(flat_images.rows, flat_images.cols, CV_64FC1);
    for(int j = 0; j<faces.size(); j++)
    {
        diff_faces.row(j) = flat_images.row(j)-mean_face;
    }
    cout<<"Mean face size = "<<mean_face.size()<<endl;
//    cout<<diff_faces<<endl;
    
    //  Now compute the covariance matrix
    //Mat covar_mat = diff_faces.t()*(diff_faces);
    Mat covar_mat = diff_faces*diff_faces.t();
    cout<<covar_mat.size()<<endl;
    
    //  Now compute the eigenvector and eigenvalues
    Mat eigenval, eigenvec;
    eigen(covar_mat, eigenval, eigenvec);
    
    cout<<eigenvec.size()<<endl;
    cout<<diff_faces.size()<<endl;
    eigenfaces = diff_faces.t()*eigenvec;
    for(int i = 0; i<eigenfaces.rows; i++)
    {
        eigenfaces.row(i) = eigenfaces.row(i)/norm(eigenfaces.row(i));
    }
//    cout<<eigenfaces<<endl;
    
//    Mat temp2;
//    Mat temp3;
//    eigenfaces.col(134).copyTo(temp2);
//    cout<<"temp2 size= "<<temp2.rows<<"x"<<temp2.cols<<endl;
//    temp3 = temp2.reshape(0,100);
//    temp3.convertTo(temp3, CV_8UC1);
////    normalize(temp3, temp3, 0, 255, NORM_MINMAX, CV_8UC1);
//    namedWindow( "mean_face", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "mean_face", temp3 );                   // Show our image inside it.
//    waitKey(0);
    
    
    
    //  Now project the image onto the eigenface matrix
    cout <<"eigenfaces"<< eigenfaces.size()<<endl;
    coeff = Mat::zeros(faces.size(), faces.size(), CV_64FC1);
    for(int i = 0; i<faces.size();i++)
    {
        for(int j = 0; j<eigenfaces.cols; j++)
        {
            double sum = 0;
            for(int k=0; k<num_eig;k++)
            {
                sum += (diff_faces.at<double>(i,j)*eigenfaces.at<double>(k, j));
                //diff_faces.row(i).dot(eigenfaces.col(j).t())
            }
            coeff.at<double>(i, j) = sum;
        }
//        cout<<coeff.row(i)<<endl;
    }
//    coeff = diff_faces.dot(eigenfaces.t());
//    cout<<coeff<<endl;

//    coeff = diff_faces*eigenfaces;
    cout << coeff.size() << endl;
//    cout << coeff << endl;
    return coeff;
}

int test(Mat &candidate){
    cout<<"Now testing"<<endl;
	Mat flat_candidate;
    //cout<<flat_candidate.size()<<endl;
    candidate.copyTo(flat_candidate);
    cout <<"TEST"<<endl;
	flat_candidate.convertTo(flat_candidate, CV_64FC1);
	flat_candidate = (flat_candidate.reshape(1, 1)); //Flatten the input candidate image
    cout<<"eigenfaces size"<<eigenfaces.size()<<endl;
	cout << "mean_face size" << mean_face.size() << endl;
	cout << "flat_candidate size" << mean_face.size() << endl;
	//mean_face.convertTo(mean_face, CV_64FC1);
    flat_candidate = flat_candidate - mean_face;
	mean_face.convertTo(mean_face, CV_8UC1);


    Mat temp;
    flat_candidate.convertTo(temp, CV_8UC1);
    temp = flat_candidate.reshape(1, 100);
    namedWindow("Display diff face", WINDOW_AUTOSIZE);   // Create a window for display.
    imshow("Display diff face", temp);
    cv::waitKey(0);
   // cout<<"Fc = "<<flat_candidate<<endl;
    cout << "fc size = " << flat_candidate.size() << endl;
	flat_candidate = flat_candidate.t();
    //Now Project
    Mat test_coefs = Mat::zeros(eigenfaces.cols,1, CV_64FC1);
//    cout << "eigen_faces type = " << eigenfaces.type() << endl;
//    cout << "fc type = " << flat_candidate.type() << endl;
//    cout<<"ef:"<<eigenfaces<<endl;
    for (int i = 0; i < eigenfaces.cols; i++) {
        double sum = 0;
        for(int k=0; k<flat_candidate.rows/2;k++)
        {
//            cout<<"operand1 : "<<flat_candidate.at<double>(1,k)<< "Operand2 =" << eigenfaces.at<double>(k, i)<<endl;
            sum += (flat_candidate.at<double>(k,1))*(eigenfaces.at<double>(k, i));
        }
        test_coefs.at<double>(i,1) = sum;
        cout<<sum<<endl;
//        test_coefs.at<double>(i,1) = flat_candidate.dot(eigenfaces.col(i).t());
    }
//    cout<<test_coef
    
//    Mat test_coefs = flat_candidate.dot(eigenfaces);
    
    //test_coefs is indexed as face,eigen_face
    cout << "coefs size = " << coeff.size() << endl;
    cout << "test_coefs size = " << test_coefs.size() << endl;
    cout<<"test_coefs"<<test_coefs<<endl;
    
    int min = INT_MAX;
    int min_id = -1;
    for (int i = 0; i < coeff.cols; i++) {
        Mat temp4 = coeff.col(i);
        if (norm(temp4, test_coefs, NORM_L2) < min) {
            min = norm(temp4, test_coefs, NORM_L2);
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