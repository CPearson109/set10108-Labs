#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <cstring>
#include <chrono>
#include <limits>  // For tracking min/max values

namespace fs = std::filesystem;

// CUDA warp size and maximum token length definitions
#define WARP_SIZE 32
#define MAX_TOKEN_LEN 32

// Macro to check for CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << static_cast<int>(err) << " \"" << cudaGetErrorString(err) << "\" \n"; \
        exit(1); \
    } \
} while(0)

// Function to read a file and store its contents in a vector
std::vector<char> read_file(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Get file size and allocate buffer
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

    // Convert characters to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

// CUDA constant memory to store the token being searched for
__constant__ char const_token[MAX_TOKEN_LEN];

// GPU kernel to calculate token occurrences with optimized reduction
__global__ void calc_token_occurrences_gpu(const char* data, int data_size, int token_len, int* block_counts)
{
    extern __shared__ char shared_mem[]; // Shared memory allocation

    int tid = threadIdx.x; // Thread ID within block
    int block_start_original = blockIdx.x * blockDim.x; // Start index for this block
    int block_end_original = (blockIdx.x + 1) * blockDim.x; // End index for this block

    // Extend block range to account for token overlap between partitions
    int block_start = block_start_original - (token_len - 1);
    int block_end = block_end_original + (token_len - 1);

    // Ensure block indices are within valid bounds
    block_start = max(0, block_start);
    block_end = min(data_size, block_end);

    int partition_size = block_end - block_start;

    // Align partition size for memory coalescing
    int partition_size_padded = (partition_size + sizeof(char4) - 1) / sizeof(char4) * sizeof(char4);

    // Shared memory regions
    char* s_data = shared_mem;
    int* s_counts = (int*)&shared_mem[partition_size_padded];

    // Load data into shared memory
    for (int s_idx = tid; s_idx < partition_size; s_idx += blockDim.x)
    {
        int global_idx = block_start + s_idx;
        if (global_idx < data_size)
            s_data[s_idx] = __ldg(&data[global_idx]);
        else
            s_data[s_idx] = 0;
    }
    __syncthreads();

    // Initialize per-thread count
    int count = 0;

    // Process data and count token occurrences
    for (int local_i = tid; local_i <= partition_size - token_len; local_i += blockDim.x)
    {
        int global_i = block_start + local_i;

        // Only count tokens within the original block range
        if (global_i >= block_start_original && global_i < block_end_original)
        {
            int match = 1;

            // Optimize token matching using integer comparisons
            if (token_len >= 4)
            {
                // Handle tokens of length 4 or more using manual byte assembly
                int num_int_comparisons = token_len / 4;
                int remaining_chars = token_len % 4;

                for (int j = 0; j < num_int_comparisons * 4; j += 4)
                {
                    // Assemble 4 bytes from s_data
                    int data_int = ((int)(unsigned char)s_data[local_i + j]) |
                        ((int)(unsigned char)s_data[local_i + j + 1] << 8) |
                        ((int)(unsigned char)s_data[local_i + j + 2] << 16) |
                        ((int)(unsigned char)s_data[local_i + j + 3] << 24);

                    // Assemble 4 bytes from const_token
                    int token_int = ((int)(unsigned char)const_token[j]) |
                        ((int)(unsigned char)const_token[j + 1] << 8) |
                        ((int)(unsigned char)const_token[j + 2] << 16) |
                        ((int)(unsigned char)const_token[j + 3] << 24);

                    if (data_int != token_int)
                    {
                        match = 0;
                        break;
                    }
                }

                // Compare remaining characters
                if (match && remaining_chars > 0)
                {
                    for (int j = num_int_comparisons * 4; j < token_len; ++j)
                    {
                        if (s_data[local_i + j] != const_token[j])
                        {
                            match = 0;
                            break;
                        }
                    }
                }
            }
            else
            {
                // For tokens shorter than 4 characters
                for (int j = 0; j < token_len; ++j)
                {
                    if (s_data[local_i + j] != const_token[j])
                    {
                        match = 0;
                        break;
                    }
                }
            }

            // Check if the token is a standalone word (not part of a larger word)
            if (match)
            {
                int iPrefix = local_i - 1;
                if (iPrefix >= 0 && s_data[iPrefix] >= 'a' && s_data[iPrefix] <= 'z')
                    match = 0;

                int iSuffix = local_i + token_len;
                if (iSuffix < partition_size&& s_data[iSuffix] >= 'a' && s_data[iSuffix] <= 'z')
                    match = 0;

                if (match)
                    count += 1; // Increment count for each valid match
            }
        }
    }

    // Each thread writes its count to shared memory
    s_counts[tid] = count;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_counts[tid] += s_counts[tid + s];
        }
        __syncthreads();
    }

    // Write the result from thread 0 of each block
    if (tid == 0)
    {
        block_counts[blockIdx.x] = s_counts[0];
    }
}

int main()
{
    // Get GPU device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    std::cout << "Running on GPU: " << deviceProp.name << std::endl;
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;

    const char* filepath = "dataset/beowulf.txt";

    // Read file data into a vector
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Allocate memory on the GPU for file data
    char* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, file_data.size() * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_data, file_data.data(), file_data.size() * sizeof(char), cudaMemcpyHostToDevice));

    // Words to search for
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };

    // Determine optimal block size using occupancy calculator
    int minGridSize = 0, blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calc_token_occurrences_gpu, 0, file_data.size()));

    int numBlocks = (file_data.size() + blockSize - 1) / blockSize;

    std::cout << "Optimal Block Size: " << blockSize << ", Number of Blocks: " << numBlocks << std::endl;

    int runs = 100;  // Number of times to run the process
    std::vector<double> times;  // Store the times for each run

    // Run the process multiple times for each word and store the times
    for (const char* word : words)
    {
        double total_time = 0.0;
        double min_time = std::numeric_limits<double>::max();
        double max_time = std::numeric_limits<double>::lowest();

        int total_occurrences = 0; // Moved outside the loop to report occurrences

        for (int i = 0; i < runs; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            int tokenLen = int(strlen(word));

            // Copy the token to constant memory on the GPU
            CUDA_CHECK(cudaMemcpyToSymbol(const_token, word, tokenLen * sizeof(char)));

            int* d_block_counts;
            CUDA_CHECK(cudaMalloc(&d_block_counts, numBlocks * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_block_counts, 0, numBlocks * sizeof(int)));

            // Calculate the maximum partition size and align it
            int max_partition_size = blockSize + 2 * (tokenLen - 1);
            int partition_size_padded = (max_partition_size + sizeof(char4) - 1) / sizeof(char4) * sizeof(char4);

            int shared_mem_size = partition_size_padded * sizeof(char) + blockSize * sizeof(int);

            // Launch the GPU kernel
            calc_token_occurrences_gpu << <numBlocks, blockSize, shared_mem_size >> > (d_data, file_data.size(), tokenLen, d_block_counts);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Retrieve the block counts from GPU memory
            std::vector<int> h_block_counts(numBlocks);
            CUDA_CHECK(cudaMemcpy(h_block_counts.data(), d_block_counts, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));

            // Sum the counts from all blocks
            int occurrences = std::accumulate(h_block_counts.begin(), h_block_counts.end(), 0);

            // Only update total_occurrences once (e.g., on the first run)
            if (i == 0)
                total_occurrences = occurrences;

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> gpu_time = end_time - start_time;

            double elapsed_time = gpu_time.count();
            total_time += elapsed_time;
            min_time = std::min(min_time, elapsed_time);
            max_time = std::max(max_time, elapsed_time);

            CUDA_CHECK(cudaFree(d_block_counts));

        }

        double avg_time = total_time / runs;

        std::cout << "\nWord: " << word << std::endl;
        std::cout << "Total Occurrences: " << total_occurrences << std::endl;
        std::cout << "Average Time: " << avg_time << " ms" << std::endl;
        std::cout << "Fastest Time: " << min_time << " ms" << std::endl;
        std::cout << "Slowest Time: " << max_time << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }

    // Free the GPU memory for file data
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
