#include <SFML/Graphics.hpp>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <iostream>
#include <fstream>
#include <chrono> // For performance timings

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fs = std::filesystem;

// Helper structure for RGBA pixels
struct rgba_t
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Helper function to load RGB data from a file
std::vector<rgba_t> load_rgb(const char* filename, int& width, int& height)
{
    int n;
    unsigned char* data = stbi_load(filename, &width, &height, &n, 4);
    if (!data)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    const rgba_t* rgbadata = reinterpret_cast<rgba_t*>(data);
    std::vector<rgba_t> vec(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature
double rgbToColorTemperature(const rgba_t& rgba)
{
    // Normalize RGB values to [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Apply gamma correction
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates
    double x = X / (X + Y + Z);
    double y = Y / (X + Y + Z);

    // Approximate color temperature using McCamy's formula
    double n = (x - 0.3320) / (0.1858 - y);
    double CCT = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;

    return CCT;
}

// Clear contents of a folder
void clear_folder(const fs::path& folder)
{
    if (fs::exists(folder) && fs::is_directory(folder))
    {
        for (const auto& entry : fs::directory_iterator(folder))
        {
            fs::remove_all(entry);
        }
    }
}

// Calculate the median color temperature from an image filename
double filename_to_median(const std::string& filename)
{
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty())
        return 0.0; // Return 0 if the image failed to load

    std::vector<double> temperatures;
    temperatures.reserve(rgbadata.size());
    std::transform(rgbadata.begin(), rgbadata.end(), std::back_inserter(temperatures), rgbToColorTemperature);
    std::sort(temperatures.begin(), temperatures.end());

    size_t size = temperatures.size();
    if (size == 0)
        return 0.0;

    double median;
    if (size % 2 == 0)
        median = 0.5 * (temperatures[size / 2 - 1] + temperatures[size / 2]);
    else
        median = temperatures[size / 2];
    return median;
}

sf::Vector2f SpriteScaleFromDimensions(const sf::Vector2u& textureSize, int screenWidth, int screenHeight)
{
    float scaleX = screenWidth / float(textureSize.x);
    float scaleY = screenHeight / float(textureSize.y);
    float scale = std::min(scaleX, scaleY);
    return { scale, scale };
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Paths for source and destination folders
    const fs::path source_folder = "images/unsorted";
    const fs::path sorted_folder = "cw2_unoptimised_sorted";

    // Create source directory if it doesn't exist
    if (!fs::exists(source_folder))
    {
        try
        {
            fs::create_directories(source_folder);
            std::cout << "Created source directory: " << source_folder << std::endl;
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Error creating source directory: " << e.what() << std::endl;
            return -1;
        }
    }

    // Create or clear sorted directory
    if (!fs::exists(sorted_folder))
    {
        try
        {
            fs::create_directories(sorted_folder);
            std::cout << "Created sorted directory: " << sorted_folder << std::endl;
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Error creating sorted directory: " << e.what() << std::endl;
            return -1;
        }
    }
    else
    {
        clear_folder(sorted_folder);
    }

    // Load image filenames from source folder
    std::vector<std::string> imageFilenames;
    for (const auto& p : fs::directory_iterator(source_folder))
    {
        if (p.is_regular_file())
        {
            imageFilenames.push_back(p.path().string());
        }
    }

    if (imageFilenames.empty())
    {
        std::cerr << "No images found in the source directory." << std::endl;
        return -1;
    }

    // Multi-threaded sorting setup
    std::mutex filenamesMutex;
    std::atomic<bool> sortingComplete(false);
    std::vector<std::string> sortedFilenames;

    // Start the sorting and copying in a separate thread
    std::thread sortingThread([&]() {
        auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

        // Copy imageFilenames under mutex protection
        std::vector<std::string> filenamesCopy;
        {
            std::lock_guard<std::mutex> lock(filenamesMutex);
            filenamesCopy = imageFilenames; // make a copy
        }

        // Compute median color temperatures in parallel
        std::vector<std::future<std::pair<std::string, double>>> futures;
        for (const auto& filename : filenamesCopy)
        {
            futures.emplace_back(std::async(std::launch::async, [filename]() {
                double median = filename_to_median(filename);
                return std::make_pair(filename, median);
                }));
        }

        // Collect the results
        std::vector<std::pair<std::string, double>> filenameMedianPairs;
        for (auto& future : futures)
        {
            filenameMedianPairs.push_back(future.get());
        }

        // Sort the filenames based on median color temperature (hottest to coldest)
        std::sort(filenameMedianPairs.begin(), filenameMedianPairs.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.second > rhs.second; // '>' for hottest to coldest
            });

        // Copy and rename the images into the sorted folder
        int index = 1;
        for (const auto& pair : filenameMedianPairs)
        {
            fs::path sourceFile = pair.first;
            fs::path destinationFile = sorted_folder / (std::to_string(index) + sourceFile.extension().string());
            try
            {
                fs::copy_file(sourceFile, destinationFile, fs::copy_options::overwrite_existing);
            }
            catch (const fs::filesystem_error& e)
            {
                std::cerr << "Error copying file " << sourceFile << ": " << e.what() << std::endl;
            }
            ++index;
        }

        // Update the sorted filenames
        std::vector<std::string> sorted;
        for (int i = 1; i < index; ++i)
        {
            fs::path filePath = sorted_folder / (std::to_string(i) + ".jpg");
            if (fs::exists(filePath))
                sorted.push_back(filePath.string());
            else
            {
                // Try other common image extensions
                for (const auto& ext : { ".png", ".jpeg", ".bmp", ".gif" })
                {
                    filePath = sorted_folder / (std::to_string(i) + ext);
                    if (fs::exists(filePath))
                    {
                        sorted.push_back(filePath.string());
                        break;
                    }
                }
            }
        }

        // Update the shared sortedFilenames under mutex protection
        {
            std::lock_guard<std::mutex> lock(filenamesMutex);
            sortedFilenames = std::move(sorted);
        }
        sortingComplete = true; // Indicate that sorting is complete

        auto end_time = std::chrono::high_resolution_clock::now(); // End timing
        std::cout << "Sorting and copying took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << " milliseconds." << std::endl;
        });

    // Define some constants
    const int gameWidth = 800;
    const int gameHeight = 600;

    int imageIndex = 0;

    // Create the window of the application
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Image Fever",
        sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);

    sf::Texture texture;
    sf::Sprite sprite;
    bool imageLoaded = false;

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

            // Arrow key handling
            if (event.type == sf::Event::KeyPressed)
            {
                if ((event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::Right) && sortingComplete)
                {
                    std::lock_guard<std::mutex> lock(filenamesMutex);
                    // Adjust the image index
                    if (event.key.code == sf::Keyboard::Left)
                        imageIndex = (imageIndex + sortedFilenames.size() - 1) % sortedFilenames.size();
                    else if (event.key.code == sf::Keyboard::Right)
                        imageIndex = (imageIndex + 1) % sortedFilenames.size();

                    // Get image filename
                    const auto& imageFilename = sortedFilenames[imageIndex];
                    // Set it as the window title
                    window.setTitle(imageFilename);
                    // Load the appropriate texture and update the sprite
                    if (texture.loadFromFile(imageFilename))
                    {
                        sprite = sf::Sprite(texture);
                        sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
                        imageLoaded = true;
                    }
                    else
                    {
                        std::cerr << "Failed to load image: " << imageFilename << std::endl;
                    }
                }
            }
        }

        // Clear the window
        window.clear(sf::Color(0, 0, 0));

        if (sortingComplete)
        {
            if (!imageLoaded)
            {
                // Load the initial image
                std::lock_guard<std::mutex> lock(filenamesMutex);
                imageFilenames = sortedFilenames; // Correctly update imageFilenames
                if (!imageFilenames.empty())
                {
                    const auto& imageFilename = imageFilenames[imageIndex];
                    if (texture.loadFromFile(imageFilename))
                    {
                        sprite = sf::Sprite(texture);
                        sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
                        window.setTitle(imageFilename);
                        imageLoaded = true;
                    }
                    else
                    {
                        std::cerr << "Failed to load image: " << imageFilename << std::endl;
                    }
                }
            }
            // Draw the sprite
            if (imageLoaded)
                window.draw(sprite);
        }

        // Display things on screen
        window.display();
    }

    // Join the sorting thread before exiting
    if (sortingThread.joinable())
        sortingThread.join();

    return EXIT_SUCCESS;
}