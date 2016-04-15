#include "eigenfaces.h"

Mat generate_flat_diff(vector<Mat> faces);
//Mat train(vector<Mat> faces);
//Mat test(Mat candidate);
void sort_mat(const Mat &input, Mat &sorted, const Mat &indices);

Mat mean_face;
Mat coefs;
Mat eigenfaces;

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
    cout<<eigenfaces<<endl;
    
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
    Mat coeff = Mat::zeros(faces.size(), faces.size(), CV_64FC1);
    for(int i = 0; i<faces.size();i++)
    {
        for(int j = 0; j<faces.size(); j++)
        {
            coeff.at<double>(i, j) = diff_faces.col(i).dot(eigenfaces.row(j).t());
        }
//        cout<<coeff.row(i)<<endl;
    }
    return eigenfaces;
}

int test(Mat &candidate, Mat &eigen_faces){
    return 0;
}