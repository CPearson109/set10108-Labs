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
#include <map>
#include <iostream>
#include <chrono>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>
#include <memory>
#include <execution>

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Define the RGBA structure before its usage
struct rgba_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Include the CUDA function declaration after defining rgba_t
extern "C" bool computeColorTemperaturesCUDA(const rgba_t* h_images, double* h_temperatures, int total_pixels);

namespace fs = std::filesystem;

// ThreadPool Class Definition
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

    auto task = std::make_shared< std::packaged_task<return_type()> >(f);

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

// Helper function to load RGB data from a file, as a contiguous array (row-major) of RGBA quadruplets
std::vector<rgba_t> load_rgb(const std::string& filename, int& width, int& height)
{
    int n;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n, 4);
    if (!data)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    const rgba_t* rgbadata = reinterpret_cast<const rgba_t*>(data);
    std::vector<rgba_t> vec;
    vec.assign(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature
inline double rgbToColorTemperature(const rgba_t& rgba) {
    // Normalize RGB values to [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Apply a gamma correction to RGB values (assumed gamma 2.2)
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

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
    double CCT = 449.0 * pow(n, 3) + 3525.0 * pow(n, 2) + 6823.3 * n + 5520.33;

    return CCT;
}

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

// Function to process images in a single-threaded manner
std::vector<std::string> singleThreadedCPU(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
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

        // Compute color temperatures
        std::vector<double> temperatures;
        temperatures.reserve(rgbadata.size());
        for (const auto& pixel : rgbadata)
        {
            temperatures.push_back(rgbToColorTemperature(pixel));
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

// Function to process images using std::async
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

            // Compute color temperatures
            std::vector<double> temperatures;
            temperatures.reserve(rgbadata.size());
            for (const auto& pixel : rgbadata)
            {
                temperatures.push_back(rgbToColorTemperature(pixel));
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

// Function to process images using ThreadPool
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

            // Compute color temperatures
            std::vector<double> temperatures;
            temperatures.reserve(rgbadata.size());
            for (const auto& pixel : rgbadata)
            {
                temperatures.push_back(rgbToColorTemperature(pixel));
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

// Function to process images using C++17 Parallel Algorithms for CPU
std::vector<std::string> parallelCPUStandardLibrary(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Pre-allocate the vector with the same size as loadedImages
    std::vector<std::pair<std::string, double>> filename_medians(loadedImages.size());

    // Use std::transform with parallel execution policy
    std::transform(std::execution::par, loadedImages.begin(), loadedImages.end(), filename_medians.begin(),
        [&](const std::pair<std::string, std::vector<rgba_t>>& pair) -> std::pair<std::string, double> {
            const std::string& filename = pair.first;
            const std::vector<rgba_t>& rgbadata = pair.second;

            double median = 0.0;
            if (!rgbadata.empty()) {
                std::vector<double> temperatures(rgbadata.size());

                // Compute color temperatures using parallel transform
                std::transform(std::execution::par, rgbadata.begin(), rgbadata.end(), temperatures.begin(),
                    [&](const rgba_t& pixel) -> double {
                        return rgbToColorTemperature(pixel);
                    });

                // Compute median
                median = calculate_median(temperatures);
            }

            return { filename, median };
        });

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

// Function to process images using C++17 Parallel Algorithms for GPU
std::vector<std::string> parallelGPUStandardLibrary(const std::vector<std::pair<std::string, std::vector<rgba_t>>>& loadedImages, double& duration)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Flatten all image data for CUDA processing
    std::vector<rgba_t> all_pixels;
    all_pixels.reserve(loadedImages.size() * 1000); // Adjust based on expected image sizes
    std::vector<int> image_pixel_counts;
    image_pixel_counts.reserve(loadedImages.size());

    for (const auto& pair : loadedImages)
    {
        if (pair.second.empty()) {
            image_pixel_counts.emplace_back(0);
            continue;
        }
        image_pixel_counts.emplace_back(static_cast<int>(pair.second.size()));
        all_pixels.insert(all_pixels.end(), pair.second.begin(), pair.second.end());
    }

    // Compute color temperatures using CUDA
    std::vector<double> all_temperatures(all_pixels.size(), 0.0);
    bool success_cuda = computeColorTemperaturesCUDA(all_pixels.data(), all_temperatures.data(), static_cast<int>(all_pixels.size()));
    if (!success_cuda)
    {
        std::cerr << "CUDA-based color temperature computation failed." << std::endl;
        return {};
    }

    // Precompute starting indices
    std::vector<int> starting_indices(loadedImages.size(), 0);
    for (size_t i = 1; i < loadedImages.size(); ++i)
    {
        starting_indices[i] = starting_indices[i - 1] + image_pixel_counts[i - 1];
    }

    // Compute medians using std::transform with parallel execution
    std::vector<std::pair<std::string, double>> filename_medians(loadedImages.size());

    std::transform(std::execution::par, loadedImages.begin(), loadedImages.end(), filename_medians.begin(),
        [&](const std::pair<std::string, std::vector<rgba_t>>& pair) -> std::pair<std::string, double> {
            const std::string& filename = pair.first;
            const std::vector<rgba_t>& rgbadata = pair.second;

            double median = 0.0;
            if (!rgbadata.empty()) {
                // Find the starting index for this image
                size_t index = &pair - &loadedImages[0]; // Get the current index

                int start = starting_indices[index];
                int img_size = image_pixel_counts[index];

                // Extract the temperatures for this image
                std::vector<double> temperatures(all_temperatures.begin() + start, all_temperatures.begin() + start + img_size);

                // Compute median
                median = calculate_median(temperatures);
            }

            return { filename, median };
        });

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

int main()
{
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Example folder to load images
    const std::string image_folder = "images/unsorted";
    if (!fs::is_directory(image_folder))
    {
        printf("Directory \"%s\" not found: please make sure it exists, and if it's a relative path, it's under your WORKING directory\n", image_folder.c_str());
        return -1;
    }
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

    // Preload all images in parallel using ThreadPool
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<std::pair<std::string, std::vector<rgba_t>>>> futures_load;
    futures_load.reserve(imageFilenames.size());

    for (const auto& filename : imageFilenames)
    {
        futures_load.emplace_back(pool.enqueue([filename]() -> std::pair<std::string, std::vector<rgba_t>> {
            int width, height;
            auto rgbadata = load_rgb(filename, width, height);
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
    std::vector<std::string> sortedFilenames_single;
    std::vector<std::string> sortedFilenames_async;
    std::vector<std::string> sortedFilenames_pool;
    std::vector<std::string> sortedFilenames_parallel_cpu;
    std::vector<std::string> sortedFilenames_parallel_gpu; // New container for C++17 Parallel Algorithms GPU

    // Containers to hold durations
    double duration_single = 0.0;
    double duration_async = 0.0;
    double duration_pool = 0.0;
    double duration_parallel_cpu = 0.0;
    double duration_parallel_gpu = 0.0; // New duration for C++17 Parallel Algorithms GPU

    // Single-Threaded CPU Sort
    sortedFilenames_single = singleThreadedCPU(loadedImages, duration_single);
    std::cout << "Single-Threaded CPU Sort Time: " << duration_single << " seconds" << std::endl;

    // Multi-Threaded CPU Sort using std::async
    sortedFilenames_async = multiThreadedCPUAsync(loadedImages, duration_async);
    std::cout << "Multi-Threaded CPU Sort using std::async Time: " << duration_async << " seconds" << std::endl;

    // Multi-Threaded CPU Sort using ThreadPool
    sortedFilenames_pool = multiThreadedCPUThreadPool(loadedImages, duration_pool);
    std::cout << "Multi-Threaded CPU Sort using ThreadPool Time: " << duration_pool << " seconds" << std::endl;

    // C++17 Parallel Algorithms-Based CPU Sort
    sortedFilenames_parallel_cpu = parallelCPUStandardLibrary(loadedImages, duration_parallel_cpu);
    std::cout << "C++17 Parallel Algorithms CPU Sort Time: " << duration_parallel_cpu << " seconds" << std::endl;

    // C++17 Parallel Algorithms-Based GPU Sort
    sortedFilenames_parallel_gpu = parallelGPUStandardLibrary(loadedImages, duration_parallel_gpu);
    std::cout << "C++17 Parallel Algorithms GPU Sort Time: " << duration_parallel_gpu << " seconds" << std::endl;

    // Determine the fastest method to display (optional)
    // For demonstration, let's choose the Parallel GPU sorted list
    // You can change this to any of the sortedFilenames_* vectors based on your preference
    std::vector<std::string> imageFilenames_sorted = sortedFilenames_parallel_gpu;

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
    // Make sure the texture fits the screen
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
                if (event.key.code == sf::Keyboard::Key::Left)
                    imageIndex = (imageIndex + imageFilenames_sorted.size() - 1) % imageFilenames_sorted.size();
                else if (event.key.code == sf::Keyboard::Key::Right)
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
