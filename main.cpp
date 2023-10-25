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
    // omp_set_num_threads(8);

    Mat img = imread(argv[1], IMREAD_ANYCOLOR);

    if (img.empty())
    {
        std::cout << " ---- Couldn't load img ----\n";
        return -1;
    }

    // cv::Mat img(3, 3, CV_32F);

    // Set pixel values
    // img.at<float>(0, 0) = 4.0f;
    // img.at<float>(0, 1) = 6.0f;
    // img.at<float>(0, 2) = 4.0f;
    // img.at<float>(1, 0) = 6.0f;
    // img.at<float>(1, 1) = 5.0f;
    // img.at<float>(1, 2) = 6.0f;
    // img.at<float>(2, 0) = 4.0f;
    // img.at<float>(2, 1) = 6.0f;
    // img.at<float>(2, 2) = 4.0f;

    Mat fimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
    img.convertTo(fimg, CV_32FC3);

    // #pragma omp parallel for collapse(2)
    for (int i = 1; i < img.rows; i++)
    {
        for (int j = 1; j < img.cols; j++)
        {
            Vec3f image = img.at<Vec3b>(i, j);
            fimg.at<Vec3f>(i, j) = image;
        }
    }

    Mat hsvimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::cvtColor(fimg, hsvimg, cv::COLOR_BGR2HSV );

    Mat lbpImg = Mat::zeros(fimg.rows, fimg.cols, CV_32FC3);

    // #pragma omp parallel for collapse(2)
    for (int i = 1; i < fimg.rows; i++)
    {
        for (int j = 1; j < fimg.cols; j++)
        {
            Vec3f center = fimg.at<cv::Vec3f>(i, j);
            float code0 = 0.0f;
            code0 += (fimg.at<Vec3f>(i - 1, j - 1)[0] > center[0]) * pow(2, 7);
            code0 += (fimg.at<Vec3f>(i - 1, j)[0] > center[0]) * pow(2, 6);
            code0 += (fimg.at<Vec3f>(i - 1, j + 1)[0] > center[0]) * pow(2, 5);
            code0 += (fimg.at<Vec3f>(i, j + 1)[0] > center[0]) * pow(2, 4);
            code0 += (fimg.at<Vec3f>(i + 1, j + 1)[0] > center[0]) * pow(2, 3);
            code0 += (fimg.at<Vec3f>(i + 1, j)[0] > center[0]) * pow(2, 2);
            code0 += (fimg.at<Vec3f>(i + 1, j - 1)[0] > center[0]) * pow(2, 1);
            code0 += (fimg.at<Vec3f>(i, j - 1)[0] > center[0]) * pow(2, 0);

            float code1 = 0.0f;
            code1 += (fimg.at<Vec3f>(i - 1, j - 1)[1] > center[1]) * pow(2, 7);
            code1 += (fimg.at<Vec3f>(i - 1, j)[1] > center[1]) * pow(2, 6);
            code1 += (fimg.at<Vec3f>(i - 1, j + 1)[1] > center[1]) * pow(2, 5);
            code1 += (fimg.at<Vec3f>(i, j + 1)[1] > center[1]) * pow(2, 4);
            code1 += (fimg.at<Vec3f>(i + 1, j + 1)[1] > center[1]) * pow(2, 3);
            code1 += (fimg.at<Vec3f>(i + 1, j)[1] > center[1]) * pow(2, 2);
            code1 += (fimg.at<Vec3f>(i + 1, j - 1)[1] > center[1]) * pow(2, 1);
            code1 += (fimg.at<Vec3f>(i, j - 1)[1] > center[1]) * pow(2, 0);

            float code2 = 0.0f;
            code2 += (fimg.at<Vec3f>(i - 1, j - 1)[2] > center[2]) * pow(2, 7);
            code2 += (fimg.at<Vec3f>(i - 1, j)[2] > center[2]) * pow(2, 6);
            code2 += (fimg.at<Vec3f>(i - 1, j + 1)[2] > center[2]) * pow(2, 5);
            code2 += (fimg.at<Vec3f>(i, j + 1)[2] > center[2]) * pow(2, 4);
            code2 += (fimg.at<Vec3f>(i + 1, j + 1)[2] > center[2]) * pow(2, 3);
            code2 += (fimg.at<Vec3f>(i + 1, j)[2] > center[2]) * pow(2, 2);
            code2 += (fimg.at<Vec3f>(i + 1, j - 1)[2] > center[2]) * pow(2, 1);
            code2 += (fimg.at<Vec3f>(i, j - 1)[2] > center[2]) * pow(2, 0);

            lbpImg.at<Vec3f>(i - 1, j - 1)[0] = code0; // blue
            lbpImg.at<Vec3f>(i - 1, j - 1)[1] = code1; // green
            lbpImg.at<Vec3f>(i - 1, j - 1)[2] = code2; // red
        }
    }

    Mat newlbpImg = Mat::zeros(hsvimg.rows, hsvimg.cols, CV_32FC3);

    for (int i = 1; i < newlbpImg.rows; i++)
    {
        for (int j = 1; j < newlbpImg.cols; j++)
        {
            edlbp(hsvimg, newlbpImg, 0, i, j);
            edlbp(hsvimg, newlbpImg, 1, i, j);
            edlbp(hsvimg, newlbpImg, 2, i, j);
        }
    }

    // imwrite("./output/bwimg.jpg", bwimg);
    // imwrite("./output/fimg.jpg", fimg);
    imwrite("./output/lbpImg.jpg", lbpImg);
    imwrite("./output/newlbpImg.jpg", newlbpImg);
    imwrite("./output/hsvimg.jpg", hsvimg);

    namedWindow("NORMAL", WINDOW_AUTOSIZE);
    imshow("NORMAL", img);
    // namedWindow("NEW LBP", WINDOW_AUTOSIZE);
    // imshow("NEW LBP", newlbpImg);
    // namedWindow("LBP", WINDOW_AUTOSIZE);
    // imshow("LBP", lbpImg);

    // namedWindow("Float", WINDOW_AUTOSIZE);
    // imshow("Float", fimg);
    // namedWindow("Black & White", WINDOW_AUTOSIZE);
    // imshow("Black & White", bwimg);
    // namedWindow("HSV Image", WINDOW_AUTOSIZE);
    // imshow("HSV Image", hsvimg);

    waitKey(0);
    destroyAllWindows();
}

void edlbp(cv::Mat img, cv::Mat newlbpImg, int index, int i, int j)
{

    float n = 0;
    float p = 0;
    p = 8;

    // comment this
    // bwimg = img;

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

    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 2;
    p = 3;
    A = (img.at<Vec3f>(i - 1, j)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i - 1, j)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "value " << one << " " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 3;
    p = 2;
    A = (img.at<Vec3f>(i - 1, j + 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i - 1, j + 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 4;
    p = 3;
    A = (img.at<Vec3f>(i, j + 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i, j + 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 5;
    p = 2;
    A = (img.at<Vec3f>(i + 1, j + 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i + 1, j + 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 6;
    p = 3;
    A = (img.at<Vec3f>(i + 1, j)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i + 1, j)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 7;
    p = 2;
    A = (img.at<Vec3f>(i + 1, j - 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i + 1, j - 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    n = 8;
    p = 3;
    A = (img.at<Vec3f>(i, j - 1)[index] - center) / (pow(2, p) - 1 - center);
    C = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
    C = floor(C);
    value = img.at<Vec3f>(i, j - 1)[index] - center;
    one = value > 0 ? 1 : 0;
    value = (one)*pow(2, (n - 1) * (1 + A * C));
    // cout << "value " << value << "\n";
    // cout << "pow : " << pow(2, (n - 1) * (1 + A * C)) << endl;
    // cout << "A: " << A << " "
    // << "C: " << C << endl;
    // cout << "code: " << code << "\n";
    // cout << endl;
    code += value;

    newlbpImg.at<Vec3f>(i - 1, j - 1)[index] = code;
}

// // cout << " " << img.rows << " " << img.cols << endl;
// Vec3b image = img.at<Vec3b>(img.rows/2, img.cols/2);
// // cout << " " << img.rows/2 << " " << img.cols/2 << endl;
// // cout << (short)image[0] << " " << (short)image[1] << " " << (short)image[2] << " ";