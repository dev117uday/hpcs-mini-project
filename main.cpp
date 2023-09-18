#include <iostream>
#include <stdio.h>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <ctime>

using namespace std;

int main(int argc, char *argv[])
{
    omp_set_num_threads(8);

    cv::Mat img = cv::imread("test.jpg", cv::IMREAD_COLOR);
    if (img.empty())
    {
        cout << " ---- Couldn't load img ----\n";
        return -1;
    }

    cv::Mat newImg = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            cv::Vec3b image = img.at<cv::Vec3b>(i, j);
            newImg.at<uchar>(i, j) = 0.114 * image[0] + 0.578 * image[1] + 0.299 * image[2];
        }
    }

    cv::Mat lbpImg = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    clock_t time_req = clock();

#pragma omp parallel for collapse(2)
    for (int i = 1; i < img.rows; i++)
    {
        for (int j = 1; j < img.cols; j++)
        {

            uchar center = newImg.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (newImg.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (newImg.at<uchar>(i - 1, j) > center) << 6;
            code |= (newImg.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (newImg.at<uchar>(i, j + 1) > center) << 4;
            code |= (newImg.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (newImg.at<uchar>(i + 1, j) > center) << 2;
            code |= (newImg.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (newImg.at<uchar>(i, j - 1) > center) << 0;
            lbpImg.at<uchar>(i - 1, j - 1) = code;
        }
    }

    time_req = clock() - time_req;
    cout << "Using pow function, it took " << (float)time_req / CLOCKS_PER_SEC << " seconds" << endl;

    bool check = imwrite("./newimg.jpg", lbpImg);
    if (check == false)
    {
        cout << "Saving the image, FAILED" << endl;
        cin.get();
        return -1;
    }

    cv::namedWindow("Image1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image1", lbpImg);
    cv::namedWindow("Image2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image2", img);
    cv::waitKey(0);
    cv::destroyWindow("Image1");
    cv::destroyWindow("Image2");
}

// cout << " " << img.rows << " " << img.cols << endl;
// cv::Vec3b image = img.at<cv::Vec3b>(img.rows/2, img.cols/2);
// cout << " " << img.rows/2 << " " << img.cols/2 << endl;
// cout << (short)image[0] << " " << (short)image[1] << " " << (short)image[2] << " ";