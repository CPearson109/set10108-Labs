#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

using namespace std;

// Kernel for simple matrix multiplication
__global__ void simple_multiply(float *output_C, unsigned int width_A, unsigned int height_A, unsigned int width_B, unsigned int height_B, const float *input_A, const float *input_B)
{
    // Get global row and column indices
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to avoid accessing invalid memory
    if (row < height_A && col < width_B) {
        float sum = 0.0f;
        // Compute the dot product of row of A and column of B
        for (unsigned int i = 0; i < width_A; ++i)
            sum += input_A[row * width_A + i] * input_B[i * width_B + col];
        // Store the result in output matrix C
        output_C[row * width_B + col] = sum;
    }
}

int main(int argc, char **argv)
{
    // Define matrix dimensions
    unsigned int width_A = 64;
    unsigned int height_A = 64;
    unsigned int width_B = 64;
    unsigned int height_B = 64;

    // Allocate host memory for matrices
    vector<float> input_A(width_A * height_A);
    vector<float> input_B(width_B * height_B);
    vector<float> output_C(width_B * height_A); // Output matrix C has dimensions height_A x width_B

    // Initialize input matrices with random float values
    for (unsigned int i = 0; i < height_A * width_A; ++i)
        input_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (unsigned int i = 0; i < height_B * width_B; ++i)
        input_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_input_A, *d_input_B, *d_output_C;
    size_t size_A = width_A * height_A * sizeof(float);
    size_t size_B = width_B * height_B * sizeof(float);
    size_t size_C = width_B * height_A * sizeof(float);

    cudaMalloc((void**)&d_input_A, size_A);
    cudaMalloc((void**)&d_input_B, size_B);
    cudaMalloc((void**)&d_output_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_input_A, input_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_B, input_B.data(), size_B, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width_B + blockDim.x - 1) / blockDim.x, (height_A + blockDim.y - 1) / blockDim.y);

    // Launch the matrix multiplication kernel
    simple_multiply<<<gridDim, blockDim>>>(d_output_C, width_A, height_A, width_B, height_B, d_input_A, d_input_B);

    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(output_C.data(), d_output_C, size_C, cudaMemcpyDeviceToHost);

    // Verify the result by comparing with CPU computation
    vector<float> reference_C(width_B * height_A);
    for (unsigned int row = 0; row < height_A; ++row) {
        for (unsigned int col = 0; col < width_B; ++col) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < width_A; ++i) {
                sum += input_A[row * width_A + i] * input_B[i * width_B + col];
            }
            reference_C[row * width_B + col] = sum;
        }
    }

        // Compare results to see if there is a difference
    float max_error = 0.0f;
    for (unsigned int i = 0; i < height_A * width_B; ++i) {
        float error = fabs(reference_C[i] - output_C[i]);
        if (error > max_error)
            max_error = error;
    }

    cout << "Max error between CPU and GPU: " << max_error << endl;

    // Free device memory
    cudaFree(d_input_A);
    cudaFree(d_input_B);
    cudaFree(d_output_C);

    cout << "Matrix multiplication completed successfully." << endl;

    return 0;
}
