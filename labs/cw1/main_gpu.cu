#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>

// Function to read the file and convert its content to lowercase
std::vector<char> read_file(const std::string& filename)
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
    std::transform(buffer.begin(), buffer.end(), buffer.begin(),
        [](unsigned char c) { return std::tolower(c); });

    return buffer;
}

__global__ void count_token_occurrences(const char* data, int data_size, const char* token, int token_length, int* count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= data_size || idx + token_length > data_size)
        return;

    // Compare token with data at position idx
    bool match = true;
    for (int i = 0; i < token_length; ++i) {
        if (data[idx + i] != token[i]) {
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

    atomicAdd(count, 1);
}

int main()
{
    // Display available files in the dataset folder
    std::string datasetFolder = "dataset";
    std::vector<std::string> txtFiles;

    std::cout << "Available text files in '" << datasetFolder << "':\n";
    for (const auto& entry : std::filesystem::directory_iterator(datasetFolder)) {
        if (entry.path().extension() == ".txt") {
            txtFiles.push_back(entry.path().filename().string());
            std::cout << " - " << entry.path().filename().string() << std::endl;
        }
    }

    // Prompt user for file selection
    std::string filename;
    std::cout << "Please enter the name of the file you wish to select: ";
    std::cin >> filename;
    std::string filepath = datasetFolder + "/" + filename;

    // Read the file
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    size_t data_size = file_data.size();
    char* d_data;
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    // Prompt user for words to search, separated by spaces
    std::string inputWords;
    std::cout << "Please enter the word(s) you wish to search for, separated by spaces: ";
    std::cin.ignore();
    std::getline(std::cin, inputWords);

    // Tokenize the input words and search each word
    std::istringstream iss(inputWords);
    std::string word;
    while (iss >> word) {
        // Convert word to lowercase to match the data
        std::transform(word.begin(), word.end(), word.begin(),
            [](unsigned char c) { return std::tolower(c); });

        int token_length = word.length();

        char* d_token;
        int* d_count;
        cudaMalloc((void**)&d_token, token_length);
        cudaMalloc((void**)&d_count, sizeof(int));

        // Copy token and reset count
        cudaMemcpy(d_token, word.c_str(), token_length, cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(int));

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start timing and launch kernel
        cudaEventRecord(start);
        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
        count_token_occurrences << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, d_token, token_length, d_count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Get time and count results
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        int occurrences;
        cudaMemcpy(&occurrences, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        // Output results
        std::cout << "Word: " << word << "\n"
            << "Occurrences: " << occurrences << "\n"
            << "Time taken: " << elapsedTime << " ms\n\n";

        // Free resources for this word
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_token);
        cudaFree(d_count);
    }

    // Free device memory
    cudaFree(d_data);

    return 0;
}
