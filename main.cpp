#include <iostream>
#include <stdio.h>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <omp.h>
#include <ctime>

using namespace std;

int main(int argc, char *argv[])
{
    omp_set_num_threads(8);

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << " ---- Couldn't load img ----\n";
        return -1;
    }

    cv::Mat bwimg = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

#pragma omp parallel for collapse(2)
    for (int i = 1; i < img.rows; i++)
    {
        for (int j = 1; j < img.cols; j++)
        {
            cv::Vec3b image = img.at<cv::Vec3b>(i, j);
            bwimg.at<uchar>(i, j) = 0.114 * image[0] + 0.578 * image[1] + 0.299 * image[2];
        }
    }

    cv::Mat lbpImg = cv::Mat::zeros(bwimg.rows, bwimg.cols, CV_8UC1);

#pragma omp parallel for collapse(2)
    for (int i = 1; i < bwimg.rows; i++)
    {
        for (int j = 1; j < bwimg.cols; j++)
        {
            uchar center = bwimg.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (bwimg.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (bwimg.at<uchar>(i - 1, j) > center) << 6;
            code |= (bwimg.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (bwimg.at<uchar>(i, j + 1) > center) << 4;
            code |= (bwimg.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (bwimg.at<uchar>(i + 1, j) > center) << 2;
            code |= (bwimg.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (bwimg.at<uchar>(i, j - 1) > center) << 0;
            lbpImg.at<uchar>(i - 1, j - 1) = code;
        }
    }

    cv::Mat newlbpImg = cv::Mat::zeros(bwimg.rows, bwimg.cols, CV_8UC1);

#pragma omp parallel for collapse(2)
    for (int i = 1; i < bwimg.rows; i++)
    {
        for (int j = 1; j < bwimg.cols; j++)
        {
            uchar center = bwimg.at<uchar>(i, j);
            uchar sumcenter = img.at<uchar>(i - 1, j - 1) + img.at<uchar>(i - 1, j) + img.at<uchar>(i - 1, j + 1) + img.at<uchar>(i, j + 1) + img.at<uchar>(i + 1, j + 1) + img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j - 1) + img.at<uchar>(i, j - 1);
            center = (uchar)(0.8 * (float)center + 0.03 * (float)sumcenter);

            unsigned char code = 0;
            code |= (bwimg.at<uchar>(i - 1, j - 1) > center) << 7 + 1;
            code |= (bwimg.at<uchar>(i - 1, j) > center) << 6 + 1;
            code |= (bwimg.at<uchar>(i - 1, j + 1) > center) << 5 + 1;
            code |= (bwimg.at<uchar>(i, j + 1) > center) << 4 + 1;
            code |= (bwimg.at<uchar>(i + 1, j + 1) > center) << 3 + 1;
            code |= (bwimg.at<uchar>(i + 1, j) > center) << 2 + 1;
            code |= (bwimg.at<uchar>(i + 1, j - 1) > center) << 1 + 1;
            code |= (bwimg.at<uchar>(i, j - 1) > center) << 1;
            newlbpImg.at<uchar>(i - 1, j - 1) = code;
        }
    }

    bool check = imwrite("./output/lbp.jpg", lbpImg);
    bool check2 = imwrite("./output/newlbp.jpg", newlbpImg);
    if (check == false || check2 ==  false)
    {
        cout << "Saving the image, FAILED" << endl;
        cin.get();
        return -1;
    }

    cv::namedWindow("NORMAL", cv::WINDOW_AUTOSIZE);
    cv::imshow("NORMAL", img);
    cv::namedWindow("NEW LBP", cv::WINDOW_AUTOSIZE);
    cv::imshow("NEW LBP", newlbpImg);
    cv::namedWindow("LBP", cv::WINDOW_AUTOSIZE);
    cv::imshow("LBP", lbpImg);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

// cout << " " << img.rows << " " << img.cols << endl;
// cv::Vec3b image = img.at<cv::Vec3b>(img.rows/2, img.cols/2);
// cout << " " << img.rows/2 << " " << img.cols/2 << endl;
// cout << (short)image[0] << " " << (short)image[1] << " " << (short)image[2] << " ";