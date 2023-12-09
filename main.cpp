#include <iostream>
#include <stdio.h>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <omp.h>
#include <ctime>

using namespace std;
using namespace cv;

void edlbp(cv::Mat img, cv::Mat newlbpImg, int index, int i, int j);

int main(int argc, char *argv[])
{
    int tnum = omp_get_num_threads();

    Mat img = imread(argv[1], IMREAD_ANYCOLOR);

    if (img.empty())
    {
        std::cout << " ---- Couldn't load img ----\n";
        return -1;
    }

    Mat fimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
    img.convertTo(fimg, CV_32FC3);
    Mat hsvimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::cvtColor(fimg, hsvimg, cv::COLOR_BGR2HSV);

    Mat newlbpImg = Mat::zeros(hsvimg.rows, hsvimg.cols, CV_32FC3);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < newlbpImg.rows; i++)
    {
        for (int j = 1; j < newlbpImg.cols; j++)
        {
            edlbp(hsvimg, newlbpImg, 0, i, j);
            edlbp(hsvimg, newlbpImg, 1, i, j);
            edlbp(hsvimg, newlbpImg, 2, i, j);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    cout << total_time.count() << endl;

    imwrite("./output/newlbpImg.jpg", newlbpImg);
}

void edlbp(cv::Mat img, cv::Mat newlbpImg, int index, int i, int j)
{

    float n = 0;
    float p = 0;
    p = 8;

    float center = img.at<Vec3f>(i, j)[index] + 0.3f;
    float sumcenter = img.at<Vec3f>(i - 1, j - 1)[index] + img.at<Vec3f>(i - 1, j)[index] + img.at<Vec3f>(i - 1, j + 1)[index] + img.at<Vec3f>(i, j + 1)[index] + img.at<Vec3f>(i + 1, j + 1)[index] + img.at<Vec3f>(i + 1, j)[index] + img.at<Vec3f>(i + 1, j - 1)[index] + img.at<Vec3f>(i, j - 1)[index];
    center = (0.5) * (center + (0.125) * sumcenter);

    float code = 0;
    float A = 0.0f;
    float C = 0.0f;
    float value = 0.0f;
    float one = 0.0f;

    n = 1;
    p = 2;
    A = (img.at<Vec3f>(i - 1, j - 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    one = (img.at<Vec3f>(i - 1, j - 1)[index] - center) > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 2;
    p = 3;
    A = (img.at<Vec3f>(i - 1, j)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i - 1, j)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 3;
    p = 2;
    A = (img.at<Vec3f>(i - 1, j + 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i - 1, j + 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 4;
    p = 3;
    A = (img.at<Vec3f>(i, j + 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i, j + 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 5;
    p = 2;
    A = (img.at<Vec3f>(i + 1, j + 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i + 1, j + 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 6;
    p = 3;
    A = (img.at<Vec3f>(i + 1, j)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i + 1, j)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 7;
    p = 2;
    A = (img.at<Vec3f>(i + 1, j - 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i + 1, j - 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    n = 8;
    p = 3;
    A = (img.at<Vec3f>(i, j - 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i, j - 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    code += value;

    newlbpImg.at<Vec3f>(i - 1, j - 1)[index] = code;
}
