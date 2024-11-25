// colour_temperature.cu
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Define the RGBA structure to match the C++ definition
struct rgba_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Optimized Kernel using double precision
__global__ void computeColorTemperaturesKernel(const rgba_t* __restrict__ d_images, double* __restrict__ d_temperatures, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // Load pixel data
    rgba_t pixel = d_images[idx];

    // Compute color temperature
    double red = pixel.r / 255.0;
    double green = pixel.g / 255.0;
    double blue = pixel.b / 255.0;

    // Apply gamma correction
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates
    double denominator = X + Y + Z;
    if (fabs(denominator) < 1e-10) {
        d_temperatures[idx] = 0.0;
        return;
    }

    double x = X / denominator;
    double y = Y / denominator;

    // Approximate color temperature using McCamy's formula
    double n = (x - 0.3320) / (0.1858 - y);
    double n2 = n * n;
    double n3 = n2 * n;
    double CCT = 449.0 * n3 + 3525.0 * n2 + 6823.3 * n + 5520.33;

    d_temperatures[idx] = CCT;
}

// Host function to compute color temperatures using CUDA
extern "C" bool computeColorTemperaturesCUDA(const rgba_t* h_images, double* h_temperatures, int total_pixels) {
    // Allocate device memory
    rgba_t* d_images = nullptr;
    double* d_temperatures_device = nullptr;
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

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy images to device asynchronously
    err = cudaMemcpyAsync(d_images, h_images, total_pixels * sizeof(rgba_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        cudaStreamDestroy(stream);
        return false;
    }

    // Define block and grid sizes
    int threads_per_block = 256;
    int blocks_per_grid = (total_pixels + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    computeColorTemperaturesKernel << <blocks_per_grid, threads_per_block, 0, stream >> > (d_images, d_temperatures_device, total_pixels);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        cudaStreamDestroy(stream);
        return false;
    }

    // Wait for GPU to finish
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA stream synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        cudaStreamDestroy(stream);
        return false;
    }

    // Copy temperatures back to host asynchronously
    err = cudaMemcpyAsync(h_temperatures, d_temperatures_device, total_pixels * sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        cudaStreamDestroy(stream);
        return false;
    }

    // Wait for copy to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA stream synchronize failed after memcpy: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_images);
        cudaFree(d_temperatures_device);
        cudaStreamDestroy(stream);
        return false;
    }

    // Free device memory and destroy stream
    cudaFree(d_images);
    cudaFree(d_temperatures_device);
    cudaStreamDestroy(stream);

    return true;
}
