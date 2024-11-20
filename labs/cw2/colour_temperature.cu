// colour_temperature.cu
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

// Define the RGBA structure to match the C++ definition
struct rgba_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Device function to convert RGB to color temperature
__device__ double rgbToColorTemperatureDevice(rgba_t rgba) {
    // Normalize RGB values to [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Apply a gamma correction to RGB values (assumed gamma 2.2)
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates
    double denominator = X + Y + Z;
    if (denominator == 0) return 0.0; // Prevent division by zero
    double x = X / denominator;
    double y = Y / denominator;

    // Approximate color temperature using McCamy's formula
    double n = (x - 0.3320) / (0.1858 - y);
    double CCT = 449.0 * pow(n, 3) + 3525.0 * pow(n, 2) + 6823.3 * n + 5520.33;

    return CCT;
}

// CUDA Kernel to compute color temperatures for all pixels
__global__ void computeColorTemperaturesKernel(const rgba_t* d_images, double* d_temperatures, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        d_temperatures[idx] = rgbToColorTemperatureDevice(d_images[idx]);
    }
}

// Host function to compute color temperatures using CUDA
extern "C" bool computeColorTemperaturesCUDA(const rgba_t* h_images, double* h_temperatures, int total_pixels) {
    // Allocate device memory
    rgba_t* d_images;
    double* d_temperatures_device;
    cudaError_t err;

    err = cudaMalloc((void**)&d_images, total_pixels * sizeof(rgba_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for images: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc((void**)&d_temperatures_device, total_pixels * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for temperatures: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        return false;
    }

    // Copy images to device
    err = cudaMemcpy(d_images, h_images, total_pixels * sizeof(rgba_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        return false;
    }

    // Define block and grid sizes
    int threads_per_block = 256;
    int blocks_per_grid = (total_pixels + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    computeColorTemperaturesKernel << <blocks_per_grid, threads_per_block >> > (d_images, d_temperatures_device, total_pixels);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        return false;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        return false;
    }

    // Copy temperatures back to host
    err = cudaMemcpy(h_temperatures, d_temperatures_device, total_pixels * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        return false;
    }

    // Free device memory
    cudaFree(d_images);
    cudaFree(d_temperatures_device);

    return true;
}
