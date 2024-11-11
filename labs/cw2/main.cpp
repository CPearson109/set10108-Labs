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
double rgbToColorTemperature(rgba_t rgba) {
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
    const fs::path sorted_folder = "images/sorted";

    // Check if source directory exists
    if (!fs::is_directory(source_folder))
    {
        std::cerr << "Directory \"" << source_folder << "\" not found: please make sure it exists." << std::endl;
        return -1;
    }

    // Create sorted directory if it doesn't exist
    if (!fs::exists(sorted_folder))
    {
        try
        {
            fs::create_directories(sorted_folder);
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Error creating sorted directory: " << e.what() << std::endl;
            return -1;
        }
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

    // Multi-threaded sorting setup
    std::mutex filenamesMutex;
    std::atomic<bool> sortingComplete(false);
    std::vector<std::string> sortedFilenames;

    // Start the sorting and copying in a separate thread
    std::thread sortingThread([&]() {
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
            return lhs.second > rhs.second; // Note the '>' for hottest to coldest
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
    }

    // Update the shared sortedFilenames under mutex protection
    {
        std::lock_guard<std::mutex> lock(filenamesMutex);
        sortedFilenames = std::move(sorted);
    }
    sortingComplete = true; // Indicate that sorting is complete
        });

    // Define some constants
    const int gameWidth = 800;
    const int gameHeight = 600;

    int imageIndex = 0;

    // Create the window of the application
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Image Fever",
        sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);

    // Load the initial image (wait until sorting is complete)
    sf::Texture texture;
    while (!sortingComplete)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Update imageFilenames from sortedFilenames
    {
        std::lock_guard<std::mutex> lock(filenamesMutex);
        imageFilenames = sortedFilenames;
    }

    // Load the first image from the sorted folder
    if (!texture.loadFromFile(imageFilenames[imageIndex]))
    {
        std::cerr << "Failed to load image: " << imageFilenames[imageIndex] << std::endl;
        return EXIT_FAILURE;
    }
    sf::Sprite sprite(texture);
    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
    window.setTitle(imageFilenames[imageIndex]);

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
                if (event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::Right)
                {
                    std::lock_guard<std::mutex> lock(filenamesMutex);
                    // Adjust the image index
                    if (event.key.code == sf::Keyboard::Left)
                        imageIndex = (imageIndex + imageFilenames.size() - 1) % imageFilenames.size();
                    else if (event.key.code == sf::Keyboard::Right)
                        imageIndex = (imageIndex + 1) % imageFilenames.size();

                    // Get image filename
                    const auto& imageFilename = imageFilenames[imageIndex];
                    // Set it as the window title
                    window.setTitle(imageFilename);
                    // Load the appropriate texture and update the sprite
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
        }

        // Clear the window
        window.clear(sf::Color(0, 0, 0));
        // Draw the sprite
        window.draw(sprite);
        // Display things on screen
        window.display();
    }

    // Join the sorting thread before exiting
    if (sortingThread.joinable())
        sortingThread.join();

    return EXIT_SUCCESS;
}