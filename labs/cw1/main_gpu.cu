#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>

// Function to read the file and convert its content to lowercase
std::vector<char> read_file(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);

    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    file.close();

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
    const char* filepath = "dataset/beowulf.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    size_t data_size = file_data.size();
    char* d_data;
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    const int num_words = sizeof(words) / sizeof(words[0]);

    int max_token_length = 0;
    for (int i = 0; i < num_words; ++i)
    {
        int token_length = strlen(words[i]);
        if (token_length > max_token_length)
            max_token_length = token_length;
    }

    char* d_token;
    cudaMalloc((void**)&d_token, max_token_length);

    int* d_count;
    cudaMalloc((void**)&d_count, sizeof(int));

    // Loop through each word and run 100 iterations
    for (int w = 0; w < num_words; ++w)
    {
        const char* word = words[w];
        int token_length = strlen(word);

        // Copy token to device
        cudaMemcpy(d_token, word, token_length, cudaMemcpyHostToDevice);

        std::vector<double> durations;
        int occurrences = 0;

        for (int run = 0; run < 1000; ++run)
        {
            // Reset count to zero
            cudaMemset(d_count, 0, sizeof(int));

            // Create CUDA events for timing each run
            cudaEvent_t word_start, word_stop;
            cudaEventCreate(&word_start);
            cudaEventCreate(&word_stop);

            // Record the start event for the word search
            cudaEventRecord(word_start);

            // Launch kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
            count_token_occurrences << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, d_token, token_length, d_count);

            // Record the stop event
            cudaEventRecord(word_stop);
            cudaEventSynchronize(word_stop);

            // Calculate elapsed time
            float word_time_ms = 0;
            cudaEventElapsedTime(&word_time_ms, word_start, word_stop);
            durations.push_back(word_time_ms);

            // Destroy events after each run
            cudaEventDestroy(word_start);
            cudaEventDestroy(word_stop);
        }

        // Copy count back to host after the final run
        cudaMemcpy(&occurrences, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        // Compute fastest, slowest, and average times
        auto minmax = std::minmax_element(durations.begin(), durations.end());
        double total_time = std::accumulate(durations.begin(), durations.end(), 0.0);
        double avg_time = total_time / durations.size();

        std::cout << "Word: " << word << "\n";
        std::cout << "Occurrences: " << occurrences << "\n";
        std::cout << "Fastest time: " << *minmax.first << " ms\n";
        std::cout << "Slowest time: " << *minmax.second << " ms\n";
        std::cout << "Average time: " << avg_time << " ms\n\n";
    }

    // Free device memory
    cudaFree(d_token);
    cudaFree(d_count);
    cudaFree(d_data);

    return 0;
}
