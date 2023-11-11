%%writefile parallel.cu

#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

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
        index = (y-1)*cols + x - 1;
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
        index = (y-1)*cols + x;
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
        index = (y-1)*cols + x + 1;
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
        index = (y+1)*cols + x + 1;
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
        index = (y+1)*cols + x;
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
        index = (y+1)*cols + x - 1;
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

        image[(y-1)*cols+x-1].x = codex;
        image[(y-1)*cols+x-1].y = codey;
        image[(y-1)*cols+x-1].z = codez;
    }
}

int main()
{
    // Read the image
    Mat img = imread("me.jpg", IMREAD_ANYCOLOR);

    Mat fimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
    img.convertTo(fimg, CV_32FC3);

    Mat image = Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::cvtColor(fimg, image, COLOR_BGR2HSV );

    // Check if the image is of type CV_32FC3
    if (image.type() == CV_32FC3)
    {
        // Get the image data pointer
        float3 *d_image;
        cudaMalloc(&d_image, image.rows * image.cols * sizeof(float3));
        cudaMemcpy(d_image, image.ptr<float3>(), image.rows * image.cols * sizeof(float3), cudaMemcpyHostToDevice);

        // Define block and grid dimensions
        dim3 block(16, 16);
        dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);

        // Launch the CUDA kernel
        processImage<<<grid, block>>>(d_image, image.rows, image.cols);

        // Copy the result back to the host
        cudaMemcpy(image.ptr<float3>(), d_image, image.rows * image.cols * sizeof(float3), cudaMemcpyDeviceToHost);

        // Free the allocated memory on the device
        cudaFree(d_image);

        // Display or save the modified image
        imwrite("m.jpg", image);
    }
    else
    {
        std::cerr << "Image is not of type CV_32FC3." << std::endl;
    }

    return 0;
}
