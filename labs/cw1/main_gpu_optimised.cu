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

    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

// GPU kernel for optimized implementation
__global__ void count_token_occurrences_optimized(const char* data, int data_size, const char* token, int token_length, int* count, int valid_start_idx, int valid_end_idx)
{
    extern __shared__ char shared_data[];

    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int shared_idx = threadIdx.x;

    if (global_idx >= data_size)
        return;

    // Load data into shared memory
    if (global_idx < data_size) {
        shared_data[shared_idx] = data[global_idx];
    }

    // Load overlap for boundary checking
    if (shared_idx < token_length - 1 && global_idx + blockDim.x < data_size) {
        shared_data[blockDim.x + shared_idx] = data[global_idx + blockDim.x];
    }

    __syncthreads();

    if (global_idx + token_length > data_size)
        return;

    // Token matching with unrolled loop
    bool match = true;
#pragma unroll
    for (int i = 0; i < token_length; ++i) {
        if (shared_data[shared_idx + i] != token[i]) {
            match = false;
            break;
        }
    }

    if (!match)
        return;

    // Check token boundaries
    if (global_idx > 0 && data[global_idx - 1] >= 'a' && data[global_idx - 1] <= 'z')
        return;

    if (global_idx + token_length < data_size && data[global_idx + token_length] >= 'a' && data[global_idx + token_length] <= 'z')
        return;

    // Check if match is within valid range
    if (global_idx < valid_start_idx || global_idx + token_length > valid_end_idx)
        return;

    atomicAdd(count, 1);
}

// Struct to hold per-chunk data
struct ChunkData {
    std::vector<char> data;
    int data_size;
    int valid_start_idx;
    int valid_end_idx;
    char* d_data;
    int* d_count;
    cudaStream_t stream;
};

// Function to split data into chunks with overlap
std::vector<ChunkData> split_data(const std::vector<char>& data, int num_chunks, int overlap)
{
    std::vector<ChunkData> chunks(num_chunks);
    size_t total_size = data.size();
    size_t base_chunk_size = total_size / num_chunks;

    for (int i = 0; i < num_chunks; ++i)
    {
        size_t start = i * base_chunk_size;
        size_t end = (i == num_chunks - 1) ? total_size : (i + 1) * base_chunk_size;

        if (i > 0)
            start -= overlap;

        if (i < num_chunks - 1)
            end += overlap;

        if (start > total_size)
            start = total_size;
        if (end > total_size)
            end = total_size;

        chunks[i].data.assign(data.begin() + start, data.begin() + end);
        chunks[i].data_size = chunks[i].data.size();

        if (i > 0)
            chunks[i].valid_start_idx = overlap;
        else
            chunks[i].valid_start_idx = 0;

        if (i < num_chunks - 1)
            chunks[i].valid_end_idx = chunks[i].data_size - overlap;
        else
            chunks[i].valid_end_idx = chunks[i].data_size;

        // Initialize device pointers and streams to nullptr
        chunks[i].d_data = nullptr;
        chunks[i].d_count = nullptr;
        chunks[i].stream = 0;
    }

    return chunks;
}

// Function to determine the maximum number of concurrent kernels
int get_max_concurrent_kernels()
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Manually set a reasonable number of streams, such as 16
    return 16;
}


// Function to perform the tuning phase
double test_configuration(int num_streams, int block_size, const std::vector<char>& file_data, const char* word, int token_length)
{
    const int NUM_TUNING_RUNS = 5;

    int overlap = token_length - 1;
    std::vector<ChunkData> chunks = split_data(file_data, num_streams, overlap);

    // Allocate and copy token to device
    char* d_token;
    cudaMalloc((void**)&d_token, token_length);
    cudaMemcpy(d_token, word, token_length, cudaMemcpyHostToDevice);

    // Allocate device memory per chunk and create streams
    for (int i = 0; i < num_streams; ++i)
    {
        ChunkData& chunk = chunks[i];

        // Create CUDA stream
        cudaStreamCreate(&chunk.stream);

        // Allocate device memory for data
        cudaMalloc((void**)&chunk.d_data, chunk.data_size);

        // Copy data to device asynchronously
        cudaMemcpyAsync(chunk.d_data, chunk.data.data(), chunk.data_size, cudaMemcpyHostToDevice, chunk.stream);

        // Allocate device memory for count
        cudaMalloc((void**)&chunk.d_count, sizeof(int));
    }

    // Synchronize to ensure data is copied
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamSynchronize(chunks[i].stream);
    }

    std::vector<double> durations;

    for (int run = 0; run < NUM_TUNING_RUNS; ++run)
    {
        auto start = std::chrono::steady_clock::now();

        // Per chunk processing
        for (int i = 0; i < num_streams; ++i)
        {
            ChunkData& chunk = chunks[i];

            // Initialize count to 0
            cudaMemsetAsync(chunk.d_count, 0, sizeof(int), chunk.stream);
        }

        // Launch kernels
        for (int i = 0; i < num_streams; ++i)
        {
            ChunkData& chunk = chunks[i];

            // Calculate grid and block sizes
            int grid_size = (chunk.data_size + block_size - 1) / block_size;

            // Shared memory size
            size_t shared_mem_size = (block_size + token_length - 1) * sizeof(char);

            // Launch kernel in stream
            count_token_occurrences_optimized << <grid_size, block_size, shared_mem_size, chunk.stream >> > (
                chunk.d_data, chunk.data_size, d_token, token_length, chunk.d_count, chunk.valid_start_idx, chunk.valid_end_idx);
        }

        // Synchronize streams
        for (int i = 0; i < num_streams; ++i)
        {
            cudaStreamSynchronize(chunks[i].stream);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        durations.push_back(elapsed.count());
    }

    // Free device memory and destroy streams
    for (int i = 0; i < num_streams; ++i)
    {
        cudaFree(chunks[i].d_data);
        cudaFree(chunks[i].d_count);
        cudaStreamDestroy(chunks[i].stream);
    }

    // Free token memory
    cudaFree(d_token);

    // Calculate average time
    double total_time = std::accumulate(durations.begin(), durations.end(), 0.0);
    double avg_time = total_time / durations.size();

    return avg_time;
}

int main()
{
    const char* filepath = "dataset/shakespeare.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    const int num_words = sizeof(words) / sizeof(words[0]);

    const int NUM_RUNS = 100;

    // Get GPU properties
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << "\n";
    std::cout << "Concurrent Kernels: " << prop.concurrentKernels << "\n";

    // Determine the optimal number of streams and block size through tuning
    std::cout << "\n=== Tuning Phase ===\n";

    std::vector<int> stream_options = { 1, 2, 4, 8, 16 };
    std::vector<int> block_size_options = { 64, 128, 256, 512 };

    // Limit the options based on GPU capabilities
    int max_streams = get_max_concurrent_kernels();
    stream_options.erase(std::remove_if(stream_options.begin(), stream_options.end(), [max_streams](int x) { return x > max_streams; }), stream_options.end());

    int max_block_size = prop.maxThreadsPerBlock;
    block_size_options.erase(std::remove_if(block_size_options.begin(), block_size_options.end(), [max_block_size](int x) { return x > max_block_size; }), block_size_options.end());

    double best_time = std::numeric_limits<double>::max();
    int best_num_streams = 1;
    int best_block_size = 64;

    const char* tuning_word = "the";
    int tuning_token_length = strlen(tuning_word);

    for (int num_streams : stream_options)
    {
        for (int block_size : block_size_options)
        {
            std::cout << "Testing num_streams = " << num_streams << ", block_size = " << block_size << "\n";
            double avg_time = test_configuration(num_streams, block_size, file_data, tuning_word, tuning_token_length);
            std::cout << "Average time: " << avg_time << " ms\n";

            if (avg_time < best_time)
            {
                best_time = avg_time;
                best_num_streams = num_streams;
                best_block_size = block_size;
            }
        }
    }

    std::cout << "\n=== Tuning Results ===\n";
    std::cout << "Optimal number of streams: " << best_num_streams << "\n";
    std::cout << "Optimal block size: " << best_block_size << "\n";

    // Main processing using the optimal parameters
    std::cout << "\n=== Main Processing ===\n";

    for (int w = 0; w < num_words; ++w)
    {
        const char* word = words[w];
        int token_length = strlen(word);

        std::cout << "Word: " << word << "\n";

        std::vector<double> durations;
        int total_occurrences = 0;

        // Split data into chunks with overlap
        int overlap = token_length - 1;
        std::vector<ChunkData> chunks = split_data(file_data, best_num_streams, overlap);

        // Allocate and copy token to device
        char* d_token;
        cudaMalloc((void**)&d_token, token_length);
        cudaMemcpy(d_token, word, token_length, cudaMemcpyHostToDevice);

        // Allocate device memory per chunk and create streams
        for (int i = 0; i < best_num_streams; ++i)
        {
            ChunkData& chunk = chunks[i];

            // Create CUDA stream
            cudaStreamCreate(&chunk.stream);

            // Allocate device memory for data
            cudaMalloc((void**)&chunk.d_data, chunk.data_size);

            // Copy data to device asynchronously
            cudaMemcpyAsync(chunk.d_data, chunk.data.data(), chunk.data_size, cudaMemcpyHostToDevice, chunk.stream);

            // Allocate device memory for count
            cudaMalloc((void**)&chunk.d_count, sizeof(int));
        }

        // Synchronize to ensure data is copied
        for (int i = 0; i < best_num_streams; ++i)
        {
            cudaStreamSynchronize(chunks[i].stream);
        }

        for (int run = 0; run < NUM_RUNS; ++run)
        {
            auto start = std::chrono::steady_clock::now();

            // Per chunk processing
            for (int i = 0; i < best_num_streams; ++i)
            {
                ChunkData& chunk = chunks[i];

                // Initialize count to 0
                cudaMemsetAsync(chunk.d_count, 0, sizeof(int), chunk.stream);
            }

            // Launch kernels
            for (int i = 0; i < best_num_streams; ++i)
            {
                ChunkData& chunk = chunks[i];

                // Calculate grid and block sizes
                int grid_size = (chunk.data_size + best_block_size - 1) / best_block_size;

                // Shared memory size
                size_t shared_mem_size = (best_block_size + token_length - 1) * sizeof(char);

                // Launch kernel in stream
                count_token_occurrences_optimized << <grid_size, best_block_size, shared_mem_size, chunk.stream >> > (
                    chunk.d_data, chunk.data_size, d_token, token_length, chunk.d_count, chunk.valid_start_idx, chunk.valid_end_idx);
            }

            // Synchronize streams
            for (int i = 0; i < best_num_streams; ++i)
            {
                cudaStreamSynchronize(chunks[i].stream);
            }

            // Copy counts back to host and sum
            int occurrences = 0;
            for (int i = 0; i < best_num_streams; ++i)
            {
                int chunk_occurrences = 0;
                cudaMemcpy(&chunk_occurrences, chunks[i].d_count, sizeof(int), cudaMemcpyDeviceToHost);
                occurrences += chunk_occurrences;
            }

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            durations.push_back(elapsed.count());

            total_occurrences = occurrences;
        }

        // Free device memory and destroy streams
        for (int i = 0; i < best_num_streams; ++i)
        {
            cudaFree(chunks[i].d_data);
            cudaFree(chunks[i].d_count);
            cudaStreamDestroy(chunks[i].stream);
        }

        // Free token memory
        cudaFree(d_token);

        // Calculate slowest, fastest, and average times
        double total_time = std::accumulate(durations.begin(), durations.end(), 0.0);
        double avg_time = total_time / durations.size();
        double fastest_time = *std::min_element(durations.begin(), durations.end());
        double slowest_time = *std::max_element(durations.begin(), durations.end());

        // Output occurrences and times
        std::cout << "Occurrences for word \"" << word << "\": " << total_occurrences << "\n";
        std::cout << "Fastest time: " << fastest_time << " ms\n";
        std::cout << "Slowest time: " << slowest_time << " ms\n";
        std::cout << "Average time with " << best_num_streams << " streams: " << avg_time << " ms\n\n";
    }

    return 0;
}
