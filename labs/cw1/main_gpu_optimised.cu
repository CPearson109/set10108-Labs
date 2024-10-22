#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

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

    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

// CPU implementation
int calc_token_occurrences_cpu(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i < int(data.size()); ++i)
    {
        if (i + tokenLen > int(data.size()))
            continue;

        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        int iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        int iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;

        ++numOccurrences;
    }
    return numOccurrences;
}

// GPU kernel for unoptimized implementation
__global__ void count_token_occurrences_gpu(const char* data, int data_size, const char* token, int token_length, int* count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= data_size)
        return;

    if (idx + token_length > data_size)
        return;

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

    int iPrefix = idx - 1;
    if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
        return;

    int iSuffix = idx + token_length;
    if (iSuffix < data_size && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
        return;

    atomicAdd(count, 1);
}

// GPU kernel for optimized implementation
__global__ void count_token_occurrences_optimized(const char* data, int data_size, const char* token, int token_length, int* count)
{
    extern __shared__ char shared_data[];

    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int block_size = blockDim.x;

    // Calculate block end, ensure it's within data_size limits
    int block_end = min(block_start + block_size, data_size);

    // Load data into shared memory with overlap for token matching
    if (global_idx < data_size)
    {
        shared_data[threadIdx.x] = data[global_idx];
    }

    // Copy extra bytes into shared memory for boundary overlap, to handle tokens spanning across block boundaries
    if (threadIdx.x < token_length - 1 && global_idx + block_size < data_size)
    {
        shared_data[block_size + threadIdx.x] = data[global_idx + block_size];
    }

    __syncthreads();

    // Ensure that the thread is working within valid data range
    if (global_idx >= data_size || global_idx + token_length > data_size)
        return;

    // Loop unrolling for token matching to reduce loop overhead
    bool match = true;
#pragma unroll
    for (int i = 0; i < token_length; ++i)
    {
        if (shared_data[threadIdx.x + i] != token[i])
        {
            match = false;
            break;
        }
    }

    if (!match)
        return;

    // Check token prefix and suffix in global memory
    if (global_idx > 0 && data[global_idx - 1] >= 'a' && data[global_idx - 1] <= 'z')
        return;

    if (global_idx + token_length < data_size && data[global_idx + token_length] >= 'a' && data[global_idx + token_length] <= 'z')
        return;

    // Atomic increment to the count
    atomicAdd(count, 1);
}

// Function to calculate optimal block and grid size based on device hardware
void calculate_optimal_block_grid_size(int data_size, int token_length, int& optimal_threadsPerBlock, int& optimal_blocksPerGrid, size_t& sharedMemSize) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimal_threadsPerBlock, count_token_occurrences_optimized, 0, 0);

    optimal_blocksPerGrid = (data_size + optimal_threadsPerBlock - 1) / optimal_threadsPerBlock;

    sharedMemSize = (optimal_threadsPerBlock + token_length - 1) * sizeof(char);

    std::cout << "Optimal Threads per Block: " << optimal_threadsPerBlock << "\n";
    std::cout << "Optimal Blocks per Grid: " << optimal_blocksPerGrid << "\n";
    std::cout << "Shared Memory Size per Block: " << sharedMemSize << " bytes\n";
}


// Class for GPU implementation
class GpuWordCounter
{
public:
    GpuWordCounter(const std::vector<char>& data)
    {
        data_size = data.size();

        cudaMalloc((void**)&d_data, data_size);
        cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice);

        max_token_length = 64;
        cudaMalloc((void**)&d_token, max_token_length);

        cudaMalloc((void**)&d_count, sizeof(int));
    }

    ~GpuWordCounter()
    {
        cudaFree(d_data);
        cudaFree(d_token);
        cudaFree(d_count);
    }

    int countOccurrences(const char* token)
    {
        int occurrences = 0;
        int token_length = strlen(token);

        cudaMemcpy(d_token, token, token_length, cudaMemcpyHostToDevice);

        cudaMemset(d_count, 0, sizeof(int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;

        count_token_occurrences_gpu << <blocksPerGrid, threadsPerBlock >> > (d_data, data_size, d_token, token_length, d_count);

        cudaDeviceSynchronize();

        cudaMemcpy(&occurrences, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        return occurrences;
    }

private:
    char* d_data;
    int data_size;

    char* d_token;
    int max_token_length;

    int* d_count;
};

// Class for optimized GPU implementation
class GpuWordCounterOptimized
{
public:
    GpuWordCounterOptimized(const std::vector<char>& data)
    {
        data_size = data.size();

        cudaMalloc((void**)&d_data, data_size);
        cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice);

        max_token_length = 64;
        cudaMalloc((void**)&d_token, max_token_length);

        cudaMalloc((void**)&d_count, sizeof(int));

        // Calculate optimal block and grid size and print it once
        calculate_optimal_block_grid_size(data_size, max_token_length, optimal_threadsPerBlock, optimal_blocksPerGrid, sharedMemSize);
    }

    ~GpuWordCounterOptimized()
    {
        cudaFree(d_data);
        cudaFree(d_token);
        cudaFree(d_count);
    }

    int countOccurrences(const char* token)
    {
        int occurrences = 0;
        int token_length = strlen(token);

        cudaMemcpy(d_token, token, token_length, cudaMemcpyHostToDevice);

        cudaMemset(d_count, 0, sizeof(int));

        // Launch kernel with previously calculated optimal settings
        count_token_occurrences_optimized << <optimal_blocksPerGrid, optimal_threadsPerBlock, sharedMemSize >> > (d_data, data_size, d_token, token_length, d_count);

        cudaDeviceSynchronize();

        cudaMemcpy(&occurrences, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        return occurrences;
    }

private:
    char* d_data;
    int data_size;

    char* d_token;
    int max_token_length;

    int* d_count;

    // Store the optimal configuration for reuse
    int optimal_threadsPerBlock;
    int optimal_blocksPerGrid;
    size_t sharedMemSize;
};

int main()
{
    const char* filepath = "dataset/shakespeare.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    const int num_words = sizeof(words) / sizeof(words[0]);

    std::cout << "\n=== CPU Implementation ===\n";
    for (int w = 0; w < num_words; ++w)
    {
        const char* word = words[w];
        std::vector<double> durations;
        int occurrences = 0;

        for (int run = 0; run < 100; ++run)
        {
            auto start = std::chrono::steady_clock::now();

            occurrences = calc_token_occurrences_cpu(file_data, word);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            durations.push_back(elapsed.count());
        }

        auto minmax = std::minmax_element(durations.begin(), durations.end());
        double total_time = std::accumulate(durations.begin(), durations.end(), 0.0);
        double avg_time = total_time / durations.size();

        std::cout << "Word: " << word << "\n";
        std::cout << "Occurrences: " << occurrences << "\n";
        std::cout << "Fastest time: " << *minmax.first << " ms\n";
        std::cout << "Slowest time: " << *minmax.second << " ms\n";
        std::cout << "Average time: " << avg_time << " ms\n\n";
    }

    std::cout << "\n=== GPU Implementation ===\n";
    GpuWordCounter gpu_counter(file_data);
    for (int w = 0; w < num_words; ++w)
    {
        const char* word = words[w];
        std::vector<double> durations;
        int occurrences = 0;

        for (int run = 0; run < 100; ++run)
        {
            auto start = std::chrono::steady_clock::now();

            occurrences = gpu_counter.countOccurrences(word);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            durations.push_back(elapsed.count());
        }

        auto minmax = std::minmax_element(durations.begin(), durations.end());
        double total_time = std::accumulate(durations.begin(), durations.end(), 0.0);
        double avg_time = total_time / durations.size();

        std::cout << "Word: " << word << "\n";
        std::cout << "Occurrences: " << occurrences << "\n";
        std::cout << "Fastest time: " << *minmax.first << " ms\n";
        std::cout << "Slowest time: " << *minmax.second << " ms\n";
        std::cout << "Average time: " << avg_time << " ms\n\n";
    }

    std::cout << "\n=== Optimized GPU Implementation ===\n";
    GpuWordCounterOptimized gpu_counter_optimized(file_data);
    for (int w = 0; w < num_words; ++w)
    {
        const char* word = words[w];
        std::vector<double> durations;
        int occurrences = 0;

        for (int run = 0; run < 100; ++run)
        {
            auto start = std::chrono::steady_clock::now();

            occurrences = gpu_counter_optimized.countOccurrences(word);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            durations.push_back(elapsed.count());
        }

        auto minmax = std::minmax_element(durations.begin(), durations.end());
        double total_time = std::accumulate(durations.begin(), durations.end(), 0.0);
        double avg_time = total_time / durations.size();

        std::cout << "Word: " << word << "\n";
        std::cout << "Occurrences: " << occurrences << "\n";
        std::cout << "Fastest time: " << *minmax.first << " ms\n";
        std::cout << "Slowest time: " << *minmax.second << " ms\n";
        std::cout << "Average time: " << avg_time << " ms\n\n";
    }

    return 0;
}
