#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>

// Function to read the file content
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

// CUDA kernel to count occurrences of words
__global__ void count_token_occurrences(const char* data, int data_size, const char* const* words, int* word_lengths, int num_words, int* counts)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= data_size) return;

    for (int w = 0; w < num_words; ++w)
    {
        int token_length = word_lengths[w];
        if (idx + token_length > data_size) continue;

        bool match = true;
        for (int i = 0; i < token_length; ++i)
        {
            if (data[idx + i] != words[w][i])
            {
                match = false;
                break;
            }
        }
        if (!match) continue;

        int iPrefix = idx - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z') continue;

        int iSuffix = idx + token_length;
        if (iSuffix < data_size && data[iSuffix] >= 'a' && data[iSuffix] <= 'z') continue;

        atomicAdd(&counts[w], 1);
    }
}

int main()
{
    // Example file path
    const char* filepath = "dataset/shakespeare.txt";

    // Read file data
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty()) return -1;

    // Copy file data to the GPU
    size_t data_size = file_data.size();
    char* d_data;
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    // Example word list
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    int num_words = sizeof(words) / sizeof(words[0]);

    // Allocate memory for word pointers and lengths on the GPU
    char** d_words;
    int* d_word_lengths;
    cudaMalloc((void**)&d_words, num_words * sizeof(char*));
    cudaMalloc((void**)&d_word_lengths, num_words * sizeof(int));

    // Allocate memory for the occurrence counts on the GPU
    int* d_counts;
    cudaMalloc((void**)&d_counts, num_words * sizeof(int));
    cudaMemset(d_counts, 0, num_words * sizeof(int));

    // Prepare word lengths and copy words to the GPU
    std::vector<int> word_lengths(num_words);
    for (int i = 0; i < num_words; ++i)
    {
        word_lengths[i] = strlen(words[i]);
        char* d_word;
        cudaMalloc((void**)&d_word, word_lengths[i]);
        cudaMemcpy(d_word, words[i], word_lengths[i], cudaMemcpyHostToDevice);
        cudaMemcpy(&d_words[i], &d_word, sizeof(char*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_word_lengths, word_lengths.data(), num_words * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for total timing
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    // Record the start event for total time
    cudaEventRecord(total_start);

    // Loop through each word, time individually, and launch kernel
    for (int w = 0; w < num_words; ++w)
    {
        // Create CUDA events for timing each word
        cudaEvent_t word_start, word_stop;
        cudaEventCreate(&word_start);
        cudaEventCreate(&word_stop);

        // Record the start event for each word
        cudaEventRecord(word_start);

        // Launch kernel for the current word
        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
        count_token_occurrences << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, d_words, d_word_lengths, num_words, d_counts);

        // Record the stop event for each word
        cudaEventRecord(word_stop);
        cudaEventSynchronize(word_stop);

        // Calculate the elapsed time for each word
        float word_time_ms = 0;
        cudaEventElapsedTime(&word_time_ms, word_start, word_stop);

        // Copy counts back to host for this word
        int occurrences = 0;
        cudaMemcpy(&occurrences, &d_counts[w], sizeof(int), cudaMemcpyDeviceToHost);

        // Print the result for the current word
        std::cout << "Found " << occurrences << " occurrences of word: " << words[w] << " in " << word_time_ms << " ms." << std::endl;

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
    for (int i = 0; i < num_words; ++i)
    {
        char* d_word;
        cudaMemcpy(&d_word, &d_words[i], sizeof(char*), cudaMemcpyHostToDevice);
        cudaFree(d_word);
    }
    cudaFree(d_words);
    cudaFree(d_word_lengths);
    cudaFree(d_counts);
    cudaFree(d_data);

    // Destroy the total events
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    return 0;
}
