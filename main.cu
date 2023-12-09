#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void edlbp(cv::Mat img, cv::Mat newlbpImg, int index, int i, int j);

__global__ void processImage(float3 *image, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {

        float n = 0;
        float p = 0;

        float center1 = image[y * cols + x].x;
        float center2 = image[y * cols + x].y;
        float center3 = image[y * cols + x].z;

        float sumcenterx = image[(y)*cols + x + 1].x + image[(y)*cols + x - 1].x + image[(y + 1) * cols + x + 1].x + image[(y + 1) * cols + x].x + image[(y + 1) * cols + x - 1].x + image[(y - 1) * cols + x + 1].x + image[(y - 1) * cols + x].x + image[(y - 1) * cols + x - 1].x;
        float sumcentery = image[(y)*cols + x + 1].y + image[(y)*cols + x - 1].y + image[(y + 1) * cols + x + 1].y + image[(y + 1) * cols + x].y + image[(y + 1) * cols + x - 1].y + image[(y - 1) * cols + x + 1].y + image[(y - 1) * cols + x].y + image[(y - 1) * cols + x - 1].y;
        float sumcenterz = image[(y)*cols + x + 1].z + image[(y)*cols + x - 1].z + image[(y + 1) * cols + x + 1].z + image[(y + 1) * cols + x].z + image[(y + 1) * cols + x - 1].z + image[(y - 1) * cols + x + 1].z + image[(y - 1) * cols + x].z + image[(y - 1) * cols + x - 1].z;

        center1 = (0.5) * (center1 + (0.125) * sumcenterx);
        center2 = (0.5) * (center2 + (0.125) * sumcentery);
        center3 = (0.5) * (center3 + (0.125) * sumcenterz);

        float codex = 0.0f, codey = 0.0f, codez = 0.0f;
        float Ax = 0.0f, Ay = 0.0f, Az = 0.0f;
        float Cx = 0.0f, Cy = 0.0f, Cz = 0.0f;
        float valuex = 0.0f, valuey = 0.0f, valuez = 0.0f;
        float onex = 0.0f, oney = 0.0f, onez = 0.0f;
        int index;

        n = 1;
        p = 2;
        index = (y - 1) * cols + x - 1;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 2;
        p = 3;
        index = (y - 1) * cols + x;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 3;
        p = 2;
        index = (y - 1) * cols + x + 1;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 4;
        p = 3;
        index = (y)*cols + x + 1;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 5;
        p = 2;
        index = (y + 1) * cols + x + 1;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 6;
        p = 3;
        index = (y + 1) * cols + x;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 7;
        p = 2;
        index = (y + 1) * cols + x - 1;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        n = 8;
        p = 3;
        index = (y)*cols + x - 1;
        Ax = (image[index].x - center1) / (pow(2, p) - 1 - center1);
        Cx = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cx = floor(Cx);
        onex = (image[index].x - center1) > 0 ? 1 : 0;
        valuex = (onex)*pow(2, (n - 1) * (1 + Ax * Cx));

        Ay = (image[index].y - center2) / (pow(2, p) - 1 - center2);
        Cy = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cy = floor(Cy);
        oney = (image[index].y - center2) > 0 ? 1 : 0;
        valuey = (oney)*pow(2, (n - 1) * (1 + Ay * Cy));

        Az = (image[index].z - center3) / (pow(2, p) - 1 - center3);
        Cz = ((-1 * (n - 1) * (n - p)) / (((p - 1) / 2) * ((p - 1) / 2)));
        Cz = floor(Cz);
        onez = (image[index].z - center3) > 0 ? 1 : 0;
        valuez = (onez)*pow(2, (n - 1) * (1 + Az * Cz));

        codex += valuex;
        codey += valuey;
        codez += valuez;

        image[(y - 1) * cols + x - 1].x = codex;
        image[(y - 1) * cols + x - 1].y = codey;
        image[(y - 1) * cols + x - 1].z = codez;
    }
}

int main(int argc, char *argv[])
{
    // Read the image
    Mat img = imread(argv[1], IMREAD_ANYCOLOR);

    Mat fimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
    img.convertTo(fimg, CV_32FC3);

    Mat image = Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::cvtColor(fimg, image, COLOR_BGR2HSV);

    // Get the image data pointer
    float3 *d_image;
    cudaMalloc(&d_image, image.rows * image.cols * sizeof(float3));
    cudaMemcpy(d_image, image.ptr<float3>(), image.rows * image.cols * sizeof(float3), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch the CUDA kernel
    processImage<<<grid, block>>>(d_image, image.rows, image.cols);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    cout << elapsed_time_ms << endl;

    // Copy the result back to the host
    cudaMemcpy(image.ptr<float3>(), d_image, image.rows * image.cols * sizeof(float3), cudaMemcpyDeviceToHost);

    // Free the allocated memory on the device
    cudaFree(d_image);

    // Display or save the modified image
    imwrite("m.jpg", image);

    Mat newlbpImg = Mat::zeros(image.rows, image.cols, CV_32FC3);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < newlbpImg.rows; i++)
    {
        for (int j = 1; j < newlbpImg.cols; j++)
        {
            edlbp(fimg, newlbpImg, 0, i, j);
            edlbp(fimg, newlbpImg, 1, i, j);
            edlbp(fimg, newlbpImg, 2, i, j);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    cout << total_time.count() << endl;

    imwrite("m1.jpg", image);

    cout << total_time.count() / elapsed_time_ms << endl;

    return 0;
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
