// colour_temperature_optimized.cu
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

// Optimized Kernel
__global__ void computeColorTemperaturesKernel(const rgba_t* __restrict__ d_images, float* __restrict__ d_temperatures, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // Load pixel data
    rgba_t pixel = d_images[idx];

    // Compute color temperature
    float red = pixel.r / 255.0f;
    float green = pixel.g / 255.0f;
    float blue = pixel.b / 255.0f;

    red = (red > 0.04045f) ? powf((red + 0.055f) / 1.055f, 2.4f) : (red / 12.92f);
    green = (green > 0.04045f) ? powf((green + 0.055f) / 1.055f, 2.4f) : (green / 12.92f);
    blue = (blue > 0.04045f) ? powf((blue + 0.055f) / 1.055f, 2.4f) : (blue / 12.92f);

    float X = red * 0.4124f + green * 0.3576f + blue * 0.1805f;
    float Y = red * 0.2126f + green * 0.7152f + blue * 0.0722f;
    float Z = red * 0.0193f + green * 0.1192f + blue * 0.9505f;

    float denominator = X + Y + Z;
    if (denominator == 0.0f) {
        d_temperatures[idx] = 0.0f;
        return;
    }

    float x = X / denominator;
    float y = Y / denominator;

    float n = (x - 0.3320f) / (0.1858f - y);
    float CCT = 449.0f * n * n * n + 3525.0f * n * n + 6823.3f * n + 5520.33f;

    d_temperatures[idx] = CCT;
}

// Host function to compute color temperatures using CUDA
extern "C" bool computeColorTemperaturesCUDA(const rgba_t * h_images, float* h_temperatures, int total_pixels) {
    // Allocate device memory
    rgba_t* d_images = nullptr;
    float* d_temperatures_device = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_images, total_pixels * sizeof(rgba_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for images: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc((void**)&d_temperatures_device, total_pixels * sizeof(float));
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
    err = cudaMemcpyAsync(h_temperatures, d_temperatures_device, total_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream);
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
