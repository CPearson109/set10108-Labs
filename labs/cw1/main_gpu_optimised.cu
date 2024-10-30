#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <cfloat> 
#include <filesystem>
#include <string>
#include <sstream>
#include <limits>

namespace fs = std::filesystem;

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
    // List all .txt files in the dataset folder
    std::string dataset_dir = "dataset";
    std::vector<std::string> txt_files;

    for (const auto& entry : fs::directory_iterator(dataset_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            txt_files.push_back(entry.path().filename().string());
        }
    }

    if (txt_files.empty()) {
        std::cerr << "No .txt files found in the dataset folder." << std::endl;
        return -1;
    }

    std::cout << "Available text files in the dataset folder:" << std::endl;
    for (size_t i = 0; i < txt_files.size(); ++i) {
        std::cout << i + 1 << ". " << txt_files[i] << std::endl;
    }

    std::cout << "Please select a file by entering the corresponding number: ";
    size_t file_choice = 0;
    std::cin >> file_choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // clear input buffer

    if (file_choice < 1 || file_choice > txt_files.size()) {
        std::cerr << "Invalid file selection." << std::endl;
        return -1;
    }

    std::string filepath = dataset_dir + "/" + txt_files[file_choice - 1];

    std::cout << "Enter the words to search for, separated by commas: ";
    std::string words_input;
    std::getline(std::cin, words_input);

    // Split the input into words
    std::vector<std::string> words;
    std::stringstream ss(words_input);
    std::string word;
    while (std::getline(ss, word, ',')) {
        // Remove leading and trailing whitespace from word
        word.erase(0, word.find_first_not_of(" \t\n\r\f\v"));
        word.erase(word.find_last_not_of(" \t\n\r\f\v") + 1);
        if (!word.empty()) {
            words.push_back(word);
        }
    }

    if (words.empty()) {
        std::cerr << "No words entered to search for." << std::endl;
        return -1;
    }

    // Read the file
    std::vector<char> file_data = read_file(filepath.c_str());
    if (file_data.empty()) return -1;

    size_t data_size = file_data.size();
    char* d_data;
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    // Query device properties
    cudaDeviceProp deviceProp;
    int device = 0;
    cudaGetDeviceProperties(&deviceProp, device);

    // Print out basic GPU specs on one line
    std::cout << "GPU: " << deviceProp.name
        << " | Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB"
        << " | Max Threads/Block: " << deviceProp.maxThreadsPerBlock << "\n";

    // Adjust variables based on hardware properties
    int warpSize = deviceProp.warpSize;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int maxBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / warpSize;

    int threadsPerBlock = warpSize * 8;
    if (threadsPerBlock > maxThreadsPerBlock)
        threadsPerBlock = maxThreadsPerBlock;

    // CUDA events for timing
    cudaEvent_t word_start, word_stop;
    cudaEventCreate(&word_start);
    cudaEventCreate(&word_stop);

    // Print adjustments made based on GPU specs
    std::cout << "Adjustments based on GPU specs:\n";
    std::cout << " - Using " << threadsPerBlock << " threads per block " << warpSize << "\n";

    for (size_t w = 0; w < words.size(); ++w) {
        const std::string& word = words[w];
        int token_length = word.length();

        // Adjust total threads needed for each word's size
        int totalThreadsNeeded = (data_size - token_length + 1);
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        blocksPerGrid = std::min(blocksPerGrid, deviceProp.multiProcessorCount * maxBlocksPerSM);

        std::cout << " - For word \"" << word << "\": blocks per grid set to " << blocksPerGrid << "\n";

        // Copy current word to constant memory
        cudaMemcpyToSymbol(d_token_const, word.c_str(), token_length);

        // Allocate counts array
        int* d_counts;
        cudaMalloc((void**)&d_counts, totalThreadsNeeded * sizeof(int));
        int* h_counts = new int[totalThreadsNeeded];

        float word_time_ms = 0;

        // Run the kernel and time it
        cudaEventRecord(word_start);
        count_token_occurrences << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, token_length, d_counts);
        cudaEventRecord(word_stop);
        cudaEventSynchronize(word_stop);

        cudaEventElapsedTime(&word_time_ms, word_start, word_stop);

        // Copy counts back to host
        cudaMemcpy(h_counts, d_counts, totalThreadsNeeded * sizeof(int), cudaMemcpyDeviceToHost);
        int occurrences = std::accumulate(h_counts, h_counts + totalThreadsNeeded, 0);

        // Print results for the current word
        std::cout << "Word: " << word << "\n";
        std::cout << "Occurrences: " << occurrences << "\n";
        std::cout << "Time taken: " << word_time_ms << " ms\n\n";

        // Clean up
        cudaFree(d_counts);
        delete[] h_counts;
    }

    // Clean up
    cudaFree(d_data);
    cudaEventDestroy(word_start);
    cudaEventDestroy(word_stop);

    return 0;
}
