#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <cfloat> // For FLT_MAX

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
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), ::tolower);
    return buffer;
}

__constant__ char d_token_const[32];

__global__ void count_token_occurrences(const char* __restrict__ data, int data_size, int token_length, int* counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int count = 0;
    for (int i = idx; i <= data_size - token_length; i += stride)
    {
        bool match = true;
        if (token_length == 3)
        {
            match = (__ldg(&data[i]) == d_token_const[0]) &&
                (__ldg(&data[i + 1]) == d_token_const[1]) &&
                (__ldg(&data[i + 2]) == d_token_const[2]);
        }
        else if (token_length == 4)
        {
            match = (__ldg(&data[i]) == d_token_const[0]) &&
                (__ldg(&data[i + 1]) == d_token_const[1]) &&
                (__ldg(&data[i + 2]) == d_token_const[2]) &&
                (__ldg(&data[i + 3]) == d_token_const[3]);
        }
        else if (token_length == 5)
        {
            match = (__ldg(&data[i]) == d_token_const[0]) &&
                (__ldg(&data[i + 1]) == d_token_const[1]) &&
                (__ldg(&data[i + 2]) == d_token_const[2]) &&
                (__ldg(&data[i + 3]) == d_token_const[3]) &&
                (__ldg(&data[i + 4]) == d_token_const[4]);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < token_length; ++j)
            {
                if (__ldg(&data[i + j]) != d_token_const[j])
                {
                    match = false;
                    break;
                }
            }
        }

        if (!match) continue;

        char prefix_char = (i > 0) ? __ldg(&data[i - 1]) : ' ';
        bool is_prefix_valid = !(prefix_char >= 'a' && prefix_char <= 'z');

        char suffix_char = (i + token_length < data_size) ? __ldg(&data[i + token_length]) : ' ';
        bool is_suffix_valid = !(suffix_char >= 'a' && suffix_char <= 'z');

        if (is_prefix_valid && is_suffix_valid) count++;
    }

    counts[idx] = count;
}

int main()
{
    // Read the file
    const char* filepath = "dataset/beowulf.txt";
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty()) return -1;

    size_t data_size = file_data.size();
    char* d_data;
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    // List of words to search for
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    int num_words = sizeof(words) / sizeof(words[0]);

    // Query device properties
    cudaDeviceProp deviceProp;
    int device = 0;
    cudaGetDeviceProperties(&deviceProp, device);

    // Adjust variables based on hardware properties
    int warpSize = deviceProp.warpSize;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int maxBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / warpSize;

    int threadsPerBlock = warpSize * 8;
    if (threadsPerBlock > maxThreadsPerBlock)
        threadsPerBlock = maxThreadsPerBlock;

    int totalThreadsNeeded = (data_size - 1);  // Placeholder size; adjusted per word below

    // Allocate counts array
    int* d_counts;
    cudaMalloc((void**)&d_counts, totalThreadsNeeded * sizeof(int));
    int* h_counts = new int[totalThreadsNeeded];

    // CUDA events for timing
    cudaEvent_t word_start, word_stop;
    cudaEventCreate(&word_start);
    cudaEventCreate(&word_stop);

    for (int w = 0; w < num_words; ++w) {
        const char* word = words[w];
        int token_length = strlen(word);

        // Adjust total threads needed for each word's size
        totalThreadsNeeded = (data_size - token_length + 1);
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        blocksPerGrid = std::min(blocksPerGrid, deviceProp.multiProcessorCount * maxBlocksPerSM);

        // Copy current word to constant memory
        cudaMemcpyToSymbol(d_token_const, word, token_length);

        float fastest_time = FLT_MAX, slowest_time = 0, total_time = 0;
        int occurrences = 0;
        int num_runs = 1000;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaEventRecord(word_start);
            count_token_occurrences << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, token_length, d_counts);
            cudaEventRecord(word_stop);
            cudaEventSynchronize(word_stop);

            float word_time_ms;
            cudaEventElapsedTime(&word_time_ms, word_start, word_stop);
            fastest_time = std::min(fastest_time, word_time_ms);
            slowest_time = std::max(slowest_time, word_time_ms);
            total_time += word_time_ms;
        }

        // Copy counts back to host
        cudaMemcpy(h_counts, d_counts, totalThreadsNeeded * sizeof(int), cudaMemcpyDeviceToHost);
        occurrences = std::accumulate(h_counts, h_counts + totalThreadsNeeded, 0);

        // Print results for the current word
        std::cout << "Word: " << word << "\n";
        std::cout << "Occurrences: " << occurrences << "\n";
        std::cout << "Fastest time: " << fastest_time << " ms\n";
        std::cout << "Slowest time: " << slowest_time << " ms\n";
        std::cout << "Average time: " << total_time / num_runs << " ms\n\n";
    }

    // Clean up
    cudaFree(d_data);
    cudaFree(d_counts);
    delete[] h_counts;
    cudaEventDestroy(word_start);
    cudaEventDestroy(word_stop);

    return 0;
}
