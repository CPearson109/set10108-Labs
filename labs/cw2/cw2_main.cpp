// cw2_main.cpp

#include <SFML/Graphics.hpp>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <future>
#include <mutex>
#include <iostream>
#include <chrono>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>
#include <memory>
#include <execution>
#include <omp.h>

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Namespace for filesystem
namespace fs = std::filesystem;

// Define the RGBA structure
struct rgba_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Forward declaration of CUDA function
extern "C" bool computeColorTemperaturesCUDA(const rgba_t * h_images, double* h_temperatures, int total_pixels);

// ============================================================================
// ThreadPool Class Definition
// ============================================================================

class ThreadPool {
public:
    // Constructor: Initialize the pool with a given number of threads
    ThreadPool(size_t threads);

    // Enqueue a task into the pool
    template<class F>
    auto enqueue(F f) -> std::future<typename std::result_of<F()>::type>;

    // Destructor: Join all threads
    ~ThreadPool();

private:
    // Vector of worker threads
    std::vector<std::thread> workers;

    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Constructor implementation
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
{
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                for (;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
            );
}

// Enqueue method implementation
template<class F>
auto ThreadPool::enqueue(F f) -> std::future<typename std::result_of<F()>::type>
{
    using return_type = typename std::result_of<F()>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(f);

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping the pool
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

// Destructor implementation
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
        worker.join();
}

// ============================================================================
// Precomputed Gamma Correction and pow(x, 2.4) Tables
// ============================================================================

static double gamma_correction_table_unoptimised[256];
static double pow24_table_unoptimised[256];

static double gamma_correction_table_optimised[256];
static double pow24_table_optimised[256];

// Function to initialize the gamma correction and pow24 tables (Unoptimised)
void initialiseTables_unoptimised() {
    for (int i = 0; i < 256; ++i) {
        double value = i / 255.0;
        gamma_correction_table_unoptimised[i] = (value <= 0.04045) ? (value / 12.92)
            : pow((value + 0.055) / 1.055, 2.4);
        pow24_table_unoptimised[i] = gamma_correction_table_unoptimised[i]; // Same as gamma correction
    }
}

// Function to initialize the gamma correction and pow24 tables (optimised)
void initialiseTables_optimised() {
    for (int i = 0; i < 256; ++i) {
        double value = i / 255.0;
        gamma_correction_table_optimised[i] = (value <= 0.04045) ? (value / 12.92)
            : pow((value + 0.055) / 1.055, 2.4);
        pow24_table_optimised[i] = gamma_correction_table_optimised[i]; // Precompute pow(x, 2.4)
    }
}

// ============================================================================
// Unoptimised Functions
// ============================================================================

// Helper function to load RGB data from a file (Unoptimised)
std::vector<rgba_t> load_rgb_unoptimised(const std::string& filename, int& width, int& height)
{
    int n;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n, 4);
    if (!data)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    const rgba_t* rgbadata = reinterpret_cast<const rgba_t*>(data);
    std::vector<rgba_t> vec(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature (Unoptimised)
inline double rgbToColourTemperature_unoptimised(const rgba_t& rgba) {

    // Normalize RGB values to [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Apply gamma correction (assumed gamma 2.2)
    red = (red <= 0.04045) ? (red / 12.92) : pow((red + 0.055) / 1.055, 2.4);
    green = (green <= 0.04045) ? (green / 12.92) : pow((green + 0.055) / 1.055, 2.4);
    blue = (blue <= 0.04045) ? (blue / 12.92) : pow((blue + 0.055) / 1.055, 2.4);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates
    double denominator = X + Y + Z;
    if (denominator == 0) return 0.0; // Prevent division by zero
    double x = X / denominator;
    double y = Y / denominator;

    // Approximate color temperature using McCamy's formula
    double n = (x - 0.3320) / (0.1858 - y);
    double CCT = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;

    return CCT;
}

// ============================================================================
// optimised Functions
// ============================================================================

// Helper function to load RGB data from a file (optimised)
std::vector<rgba_t> load_rgb_optimised(const std::string& filename, int& width, int& height)
{
    int n;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n, 4);
    if (!data)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    const rgba_t* rgbadata = reinterpret_cast<const rgba_t*>(data);
    std::vector<rgba_t> vec(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature (optimised)
inline double rgbToColourTemperature_optimised(const rgba_t& rgba) {

    // Normalize RGB values to [0, 1] using precomputed gamma correction table
    double red = gamma_correction_table_optimised[rgba.r];
    double green = gamma_correction_table_optimised[rgba.g];
    double blue = gamma_correction_table_optimised[rgba.b];

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates
    double denominator = X + Y + Z;
    if (denominator == 0) return 0.0; // Prevent division by zero
    double x = X / denominator;
    double y = Y / denominator;

    // Approximate color temperature using McCamy's formula
    double n = (x - 0.3320) / (0.1858 - y);
    double CCT = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;

    return CCT;
}

// ============================================================================
// Median Calculation
// ============================================================================

// Calculate the median from a vector of temperatures
double calculate_median(std::vector<double>& temperatures) {
    size_t size = temperatures.size();
    if (size == 0) return 0.0;
    size_t mid = size / 2;
    std::nth_element(temperatures.begin(), temperatures.begin() + mid, temperatures.end());
    if (size % 2 == 0) {
        double mid1 = temperatures[mid - 1];
        double mid2 = temperatures[mid];
        return (mid1 + mid2) / 2.0;
    }
    else {
        return temperatures[mid];
    }
}

// ============================================================================
// Processing Functions
// ============================================================================

// Function to process images in a single-threaded manner (Unoptimised)
std::vector<std::string> singleThreadedCPU_unoptimised(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, double>> filename_medians;
    filename_medians.reserve(loadedImages.size());

    for (const auto& pair : loadedImages)
    {
        const auto& filename = pair.first;
        const auto& rgbadata = pair.second;

        if (rgbadata.empty()) {
            filename_medians.emplace_back(filename, 0.0);
            continue;
        }

        // Compute colour temperatures
        std::vector<double> temperatures;
        temperatures.reserve(rgbadata.size());
        for (const auto& pixel : rgbadata)
        {
            temperatures.push_back(rgbToColourTemperature_unoptimised(pixel));
        }

        // Compute median
        double median = calculate_median(temperatures);
        filename_medians.emplace_back(filename, median);
    }

    // Sort based on median
    std::sort(filename_medians.begin(), filename_medians.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) -> bool {
            return a.second < b.second;
        });

    // Extract sorted filenames
    std::vector<std::string> sortedFilenames;
    sortedFilenames.reserve(filename_medians.size());
    for (const auto& pair : filename_medians)
    {
        sortedFilenames.push_back(pair.first);
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    return sortedFilenames;
}

// Function to process images in a single-threaded manner (optimised)
std::vector<std::string> singleThreadedCPU_optimised(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, double>> filename_medians;
    filename_medians.reserve(loadedImages.size());

    for (const auto& pair : loadedImages)
    {
        const auto& filename = pair.first;
        const auto& rgbadata = pair.second;

        if (rgbadata.empty()) {
            filename_medians.emplace_back(filename, 0.0);
            continue;
        }

        // Compute colour temperatures
        std::vector<double> temperatures;
        temperatures.reserve(rgbadata.size());
        for (const auto& pixel : rgbadata)
        {
            temperatures.push_back(rgbToColourTemperature_optimised(pixel));
        }

        // Compute median
        double median = calculate_median(temperatures);
        filename_medians.emplace_back(filename, median);
    }

    // Sort based on median
    std::sort(filename_medians.begin(), filename_medians.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) -> bool {
            return a.second < b.second;
        });

    // Extract sorted filenames
    std::vector<std::string> sortedFilenames;
    sortedFilenames.reserve(filename_medians.size());
    for (const auto& pair : filename_medians)
    {
        sortedFilenames.push_back(pair.first);
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    return sortedFilenames;
}

// Function to process images using std::async (optimised)
std::vector<std::string> multiThreadedCPUAsync(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<std::pair<std::string, double>>> futures;
    futures.reserve(loadedImages.size());

    for (const auto& pair : loadedImages)
    {
        const auto& filename = pair.first;
        const auto& rgbadata = pair.second;

        futures.emplace_back(std::async(std::launch::async, [filename, rgbadata]() -> std::pair<std::string, double> {
            if (rgbadata.empty()) {
                return { filename, 0.0 };
            }

        // Compute colour temperatures
        std::vector<double> temperatures;
        temperatures.reserve(rgbadata.size());
        for (const auto& pixel : rgbadata)
        {
            temperatures.push_back(rgbToColourTemperature_optimised(pixel));
        }

        // Compute median
        double median = calculate_median(temperatures);
        return { filename, median };
            }));
    }

    // Collect results
    std::vector<std::pair<std::string, double>> filename_medians;
    filename_medians.reserve(loadedImages.size());
    for (auto& fut : futures)
    {
        filename_medians.emplace_back(fut.get());
    }

    // Sort based on median
    std::sort(filename_medians.begin(), filename_medians.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) -> bool {
            return a.second < b.second;
        });

    // Extract sorted filenames
    std::vector<std::string> sortedFilenames;
    sortedFilenames.reserve(filename_medians.size());
    for (const auto& pair : filename_medians)
    {
        sortedFilenames.push_back(pair.first);
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    return sortedFilenames;
}

// Function to process images using ThreadPool (optimised)
std::vector<std::string> multiThreadedCPUThreadPool(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<std::pair<std::string, double>>> futures;
    futures.reserve(loadedImages.size());

    for (const auto& pair : loadedImages)
    {
        const auto& filename = pair.first;
        const auto& rgbadata = pair.second;

        futures.emplace_back(pool.enqueue([filename, rgbadata]() -> std::pair<std::string, double> {
            if (rgbadata.empty()) {
                return { filename, 0.0 };
            }

        // Compute colour temperatures
        std::vector<double> temperatures;
        temperatures.reserve(rgbadata.size());
        for (const auto& pixel : rgbadata)
        {
            temperatures.push_back(rgbToColourTemperature_optimised(pixel));
        }

        // Compute median
        double median = calculate_median(temperatures);
        return { filename, median };
            }));
    }

    // Collect results
    std::vector<std::pair<std::string, double>> filename_medians;
    filename_medians.reserve(loadedImages.size());
    for (auto& fut : futures)
    {
        filename_medians.emplace_back(fut.get());
    }

    // Sort based on median
    std::sort(filename_medians.begin(), filename_medians.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) -> bool {
            return a.second < b.second;
        });

    // Extract sorted filenames
    std::vector<std::string> sortedFilenames;
    sortedFilenames.reserve(filename_medians.size());
    for (const auto& pair : filename_medians)
    {
        sortedFilenames.push_back(pair.first);
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    return sortedFilenames;
}

// optimised Function: Parallel CPU Sort using C++17 Parallel Algorithms with OpenMP and SIMD
std::vector<std::string> parallelCPUStandardLibrary(
    const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages,
    double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    size_t num_images = loadedImages.size();

    // Pre-allocate the vector with the same size as loadedImages
    std::vector<std::pair<std::string, double>> filename_medians(num_images);

    // Use OpenMP to parallelize the outer loop over images
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_images; ++i)
    {
        const std::string& filename = loadedImages[i].first;
        const std::vector<rgba_t>& rgbadata = loadedImages[i].second;

        double median = 0.0;
        if (!rgbadata.empty()) {
            size_t pixel_count = rgbadata.size();

            // Use a fixed-size array to store temperatures to avoid dynamic allocation
            std::unique_ptr<double[]> temperatures(new double[pixel_count]);

            // Vectorize the loop using OpenMP SIMD directive
#pragma omp simd aligned(temperatures, rgbadata: 32)
            for (size_t j = 0; j < pixel_count; ++j) {
                temperatures[j] = rgbToColourTemperature_optimised(rgbadata[j]);
            }

            // Compute median
            if (pixel_count > 0) {
                size_t mid = pixel_count / 2;
                std::nth_element(temperatures.get(), temperatures.get() + mid, temperatures.get() + pixel_count);
                if (pixel_count % 2 == 0) {
                    median = (temperatures[mid - 1] + temperatures[mid]) / 2.0;
                }
                else {
                    median = temperatures[mid];
                }
            }
        }

        // Store the result
        filename_medians[i] = { filename, median };
    }

    // Sort based on median using C++17 Parallel Algorithms
    std::sort(std::execution::par_unseq, filename_medians.begin(), filename_medians.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    // Extract sorted filenames
    std::vector<std::string> sortedFilenames(num_images);
    std::transform(std::execution::par, filename_medians.begin(), filename_medians.end(),
        sortedFilenames.begin(),
        [](const auto& pair) { return pair.first; });

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    return sortedFilenames;
}

// Function to process images using CUDA (optimised)
std::vector<std::string> multiThreadedCPUCUDAMethod(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, double>> filename_medians;
    filename_medians.reserve(loadedImages.size());

    for (const auto& pair : loadedImages)
    {
        const auto& filename = pair.first;
        const auto& rgbadata = pair.second;

        if (rgbadata.empty()) {
            filename_medians.emplace_back(filename, 0.0);
            continue;
        }

        int total_pixels = static_cast<int>(rgbadata.size());
        std::vector<double> temperatures(total_pixels);

        bool success = computeColorTemperaturesCUDA(rgbadata.data(), temperatures.data(), total_pixels);
        if (!success) {
            std::cerr << "CUDA processing failed for image: " << filename << std::endl;
            filename_medians.emplace_back(filename, 0.0);
            continue;
        }

        double median = calculate_median(temperatures);
        filename_medians.emplace_back(filename, median);
    }

    // Sort based on median
    std::sort(filename_medians.begin(), filename_medians.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) -> bool {
            return a.second < b.second;
        });

    // Extract sorted filenames
    std::vector<std::string> sortedFilenames;
    sortedFilenames.reserve(filename_medians.size());
    for (const auto& pair : filename_medians)
    {
        sortedFilenames.push_back(pair.first);
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    return sortedFilenames;
}

// ============================================================================
// Verification Function
// ============================================================================

// Helper function to verify that all sorted lists are identical
bool verify_sorting_results(const std::vector<std::pair<std::string, std::vector<std::string>>>& sorted_results)
{
    if (sorted_results.empty()) return true;

    const std::vector<std::string>& reference = sorted_results[0].second;
    bool all_match = true;

    for (size_t i = 1; i < sorted_results.size(); ++i)
    {
        const auto& current = sorted_results[i].second;
        if (current != reference)
        {
            std::cerr << "Discrepancy found in method: " << sorted_results[i].first << std::endl;
            all_match = false;
        }
    }

    return all_match;
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Example folder to load images
    const std::string image_folder = "imagesLarge";
    if (!fs::is_directory(image_folder))
    {
        printf("Directory \"%s\" not found: please make sure it exists, and if it's a relative path, it's under your WORKING directory\n", image_folder.c_str());
        return -1;
    }

    // Collect image filenames
    std::vector<std::string> imageFilenames;
    for (auto& p : fs::directory_iterator(image_folder))
    {
        if (p.is_regular_file()) {
            imageFilenames.push_back(p.path().u8string());
        }
    }

    if (imageFilenames.empty())
    {
        std::cerr << "No images found in the directory: " << image_folder << std::endl;
        return -1;
    }

    // Initialize gamma correction and pow24 tables for both unoptimised and optimised functions
    initialiseTables_unoptimised();
    initialiseTables_optimised();

    // Preload all images in parallel using ThreadPool
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<std::pair<std::string, std::vector<rgba_t>>>> futures_load;
    futures_load.reserve(imageFilenames.size());

    for (const auto& filename : imageFilenames)
    {
        futures_load.emplace_back(pool.enqueue([filename]() -> std::pair<std::string, std::vector<rgba_t>> {
            int width, height;
        // Choose which load function to use here
        // For preloading, we'll use the optimised version
        auto rgbadata = load_rgb_optimised(filename, width, height);
        return { filename, std::move(rgbadata) };
            }));
    }

    // Collect loaded images
    std::vector<std::pair<std::string, std::vector<rgba_t>>> loadedImages;
    loadedImages.reserve(imageFilenames.size());
    for (auto& fut : futures_load)
    {
        loadedImages.emplace_back(fut.get());
    }

    // Containers to hold sorted filenames for each method
    std::vector<std::string> sortedFilenames_single_unoptimised;       // Single Thread Unoptimised
    std::vector<std::string> sortedFilenames_single_optimised;         // Single Thread optimised
    std::vector<std::string> sortedFilenames_async;        // Async
    std::vector<std::string> sortedFilenames_pool;         // Thread Pool
    std::vector<std::string> sortedFilenames_parallel_cpu; // C++17 Parallel Algorithms CPU
    std::vector<std::string> sortedFilenames_cuda;         // CUDA

    // Containers to hold durations
    double duration_single_unoptimised = 0.0;       // Single Thread Unoptimised
    double duration_single_optimised = 0.0;         // Single Thread optimised
    double duration_async = 0.0;        // Async
    double duration_pool = 0.0;         // Thread Pool
    double duration_parallel_cpu = 0.0; // C++17 Parallel Algorithms CPU
    double duration_cuda = 0.0;         // CUDA

    // Single-Threaded CPU Sort (Unoptimised)
    sortedFilenames_single_unoptimised = singleThreadedCPU_unoptimised(loadedImages, duration_single_unoptimised);
    std::cout << "Single-Threaded CPU Sort (Unoptimised) Time: " << duration_single_unoptimised << " seconds" << std::endl;

    // Single-Threaded CPU Sort (optimised)
    sortedFilenames_single_optimised = singleThreadedCPU_optimised(loadedImages, duration_single_optimised);
    std::cout << "Single-Threaded CPU Sort (optimised) Time: " << duration_single_optimised << " seconds" << std::endl;

    // Multi-Threaded CPU Sort using std::async (optimised)
    sortedFilenames_async = multiThreadedCPUAsync(loadedImages, duration_async);
    std::cout << "Multi-Threaded CPU Sort using std::async Time: " << duration_async << " seconds" << std::endl;

    // Multi-Threaded CPU Sort using ThreadPool (optimised)
    sortedFilenames_pool = multiThreadedCPUThreadPool(loadedImages, duration_pool);
    std::cout << "Multi-Threaded CPU Sort using ThreadPool Time: " << duration_pool << " seconds" << std::endl;

    // C++17 Parallel Algorithms-Based CPU Sort (optimised)
    sortedFilenames_parallel_cpu = parallelCPUStandardLibrary(loadedImages, duration_parallel_cpu);
    std::cout << "C++17 Parallel Algorithms CPU Sort Time: " << duration_parallel_cpu << " seconds" << std::endl;

    // GPU-Accelerated CPU Sort using CUDA (optimised)
    sortedFilenames_cuda = multiThreadedCPUCUDAMethod(loadedImages, duration_cuda);
    std::cout << "GPU-Accelerated CPU Sort using CUDA Time: " << duration_cuda << " seconds" << std::endl;

    // Organize all sorted results with their method names
    std::vector<std::pair<std::string, std::vector<std::string>>> sorted_results = {
        { "Single-Threaded CPU (Unoptimised)", sortedFilenames_single_unoptimised },
        { "Single-Threaded CPU (optimised)", sortedFilenames_single_optimised },
        { "Multi-Threaded CPU using std::async", sortedFilenames_async },
        { "Multi-Threaded CPU using ThreadPool", sortedFilenames_pool },
        { "C++17 Parallel Algorithms CPU", sortedFilenames_parallel_cpu },
        { "GPU-Accelerated CUDA", sortedFilenames_cuda }
    };

    // Verify that all sorted lists are identical
    bool all_match = verify_sorting_results(sorted_results);

    if (all_match)
    {
        std::cout << "\nAll sorting methods produced identical results." << std::endl;
    }
    else
    {
        std::cout << "\nSome sorting methods produced different results. Please check the above discrepancies." << std::endl;
    }

    // Determine the fastest method to display
    double min_duration = duration_single_unoptimised;
    std::vector<std::string>* fastest_sorted_filenames = &sortedFilenames_single_unoptimised;
    std::string fastest_method = "Single-Threaded CPU (Unoptimised)";

    if (duration_single_optimised < min_duration) {
        min_duration = duration_single_optimised;
        fastest_sorted_filenames = &sortedFilenames_single_optimised;
        fastest_method = "Single-Threaded CPU (optimised)";
    }
    if (duration_async < min_duration) {
        min_duration = duration_async;
        fastest_sorted_filenames = &sortedFilenames_async;
        fastest_method = "Multi-Threaded CPU using std::async";
    }
    if (duration_pool < min_duration) {
        min_duration = duration_pool;
        fastest_sorted_filenames = &sortedFilenames_pool;
        fastest_method = "Multi-Threaded CPU using ThreadPool";
    }
    if (duration_parallel_cpu < min_duration) {
        min_duration = duration_parallel_cpu;
        fastest_sorted_filenames = &sortedFilenames_parallel_cpu;
        fastest_method = "C++17 Parallel Algorithms CPU";
    }
    if (duration_cuda < min_duration) {
        min_duration = duration_cuda;
        fastest_sorted_filenames = &sortedFilenames_cuda;
        fastest_method = "GPU-Accelerated CUDA";
    }

    std::cout << "\nFastest Method: " << fastest_method << " with " << min_duration << " seconds." << std::endl;

    // Choose the fastest sorted list for display
    std::vector<std::string> imageFilenames_sorted = *fastest_sorted_filenames;

    // Define some constants
    const int gameWidth = 800;
    const int gameHeight = 600;

    int imageIndex = 0;

    // Create the window of the application
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Image Fever - Concurrent Sort",
        sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);

    // Load the first image
    sf::Texture texture;
    if (!texture.loadFromFile(imageFilenames_sorted[imageIndex]))
    {
        std::cerr << "Failed to load image: " << imageFilenames_sorted[imageIndex] << std::endl;
        return EXIT_FAILURE;
    }
    sf::Sprite sprite(texture);

    // Function to scale sprite based on texture and window size
    auto SpriteScaleFromDimensions = [](const sf::Vector2u& textureSize, int screenWidth, int screenHeight) -> sf::Vector2f {
        float scaleX = static_cast<float>(screenWidth) / static_cast<float>(textureSize.x);
        float scaleY = static_cast<float>(screenHeight) / static_cast<float>(textureSize.y);
        float scale = std::min(scaleX, scaleY);
        return { scale, scale };
    };
    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));

    // Main loop
    while (window.isOpen())
    {
        // Handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Window closed or escape key pressed: exit
            if ((event.type == sf::Event::Closed) ||
                ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape)))
            {
                window.close();
                break;
            }

            // Window size changed, adjust view appropriately
            if (event.type == sf::Event::Resized)
            {
                sf::View view;
                view.setSize(gameWidth, gameHeight);
                view.setCenter(gameWidth / 2.f, gameHeight / 2.f);
                window.setView(view);
            }

            // Arrow key handling!
            if (event.type == sf::Event::KeyPressed)
            {
                // Adjust the image index
                if (event.key.code == sf::Keyboard::Left)
                    imageIndex = (imageIndex + imageFilenames_sorted.size() - 1) % imageFilenames_sorted.size();
                else if (event.key.code == sf::Keyboard::Right)
                    imageIndex = (imageIndex + 1) % imageFilenames_sorted.size();

                // Get image filename
                const auto& imageFilename = imageFilenames_sorted[imageIndex];

                // Set it as the window title 
                window.setTitle(imageFilename);

                // ... and load the appropriate texture, and put it in the sprite
                if (texture.loadFromFile(imageFilename))
                {
                    sprite = sf::Sprite(texture);
                    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
                }
                else
                {
                    std::cerr << "Failed to load image: " << imageFilename << std::endl;
                }
            }
        }

        // Clear the window
        window.clear(sf::Color(0, 0, 0));

        // Draw the sprite
        window.draw(sprite);

        // Display things on screen
        window.display();
    }

    return EXIT_SUCCESS;
}
