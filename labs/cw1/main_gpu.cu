#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>

std::vector<char> read_file(const char* filename)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Move the file cursor to the end of the file to get its size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    // Return the file cursor to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Create a vector of the same size as the file to hold the content
    std::vector<char> buffer(fileSize);

    // Read the entire file into the vector
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    // Close the file
    file.close();

    // Output the number of bytes read
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    // Convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

__global__ void count_token_occurrences(const char* data, int data_size, const char* token, int token_length, int* count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= data_size)
        return;

    // Check if token matches at position idx
    if (idx + token_length > data_size)
        return;

    // Compare token with data at position idx
    bool match = true;
    for (int i = 0; i < token_length; ++i)
    {
        if (data[idx + i] != token[i])
        {
            match = false;
            break;
        }
    }
    if (!match)
        return;

    // Test prefix character
    int iPrefix = idx - 1;
    if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
        return;

    // Test suffix character
    int iSuffix = idx + token_length;
    if (iSuffix < data_size && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
        return;

    // If we reach here, it's a valid occurrence
    atomicAdd(count, 1);
}

int main()
{
    // Example chosen file
    const char* filepath = "dataset/shakespeare.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Copy the file data to the GPU
    size_t data_size = file_data.size();
    char* d_data;
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    // Example word list
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };

    // Create CUDA events for total timing
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    // Record the start event for the total time
    cudaEventRecord(total_start);

    // Loop through each word and find its occurrences
    for (const char* word : words)
    {
        int occurrences = 0;
        int token_length = strlen(word);

        // Allocate memory for token on device
        char* d_token;
        cudaMalloc((void**)&d_token, token_length);
        cudaMemcpy(d_token, word, token_length, cudaMemcpyHostToDevice);

        // Allocate memory for count
        int* d_count;
        cudaMalloc((void**)&d_count, sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));

        // Create CUDA events for timing each word
        cudaEvent_t word_start, word_stop;
        cudaEventCreate(&word_start);
        cudaEventCreate(&word_stop);

        // Record the start event for each word
        cudaEventRecord(word_start);

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
        count_token_occurrences << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, d_token, token_length, d_count);

        // Record the stop event for each word
        cudaEventRecord(word_stop);
        cudaEventSynchronize(word_stop);

        // Calculate the elapsed time for each word
        float word_time_ms = 0;
        cudaEventElapsedTime(&word_time_ms, word_start, word_stop);

        // Copy count back to host
        cudaMemcpy(&occurrences, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Found " << occurrences << " occurrences of word: " << word << " in " << word_time_ms << " ms." << std::endl;

        // Free device memory for each word
        cudaFree(d_token);
        cudaFree(d_count);

        // Destroy the word events
        cudaEventDestroy(word_start);
        cudaEventDestroy(word_stop);
    }

    // Record the stop event for total time
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);

    // Calculate the total elapsed time
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, total_start, total_stop);

    std::cout << "Total time to find all occurrences: " << total_time_ms << " ms." << std::endl;

    // Free device memory
    cudaFree(d_data);

    // Destroy the total events
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    return 0;
}
