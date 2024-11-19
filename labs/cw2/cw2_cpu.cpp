#include <SFML/Graphics.hpp>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm> // For std::sort and std::transform
#include <cstdio>    // For printf

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fs = std::filesystem;

// Helper structure for RGBA pixels (a is safe to ignore for this coursework)
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
        printf("Failed to load image: %s\n", filename);
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

// Calculate the median from an image filename
double filename_to_median(const std::string& filename)
{
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty())
        return 0.0;

    std::vector<double> temperatures;
    std::transform(rgbadata.begin(), rgbadata.end(), std::back_inserter(temperatures), rgbToColorTemperature);
    std::sort(temperatures.begin(), temperatures.end());

    size_t size = temperatures.size();
    if (size % 2 == 0)
        return 0.5 * (temperatures[size / 2 - 1] + temperatures[size / 2]);
    else
        return temperatures[size / 2];
}

// Clear contents of a folder if it exists
void clear_folder(const std::string& folder_path)
{
    fs::path folder = folder_path;
    if (fs::exists(folder) && fs::is_directory(folder))
    {
        for (const auto& entry : fs::directory_iterator(folder))
        {
            fs::remove_all(entry);
        }
    }
}

// Sort images based on median color temperature and save them to a new folder
void sort_and_save_images(const std::vector<std::string>& filenames, const std::string& destination_folder)
{
    // Create a vector of pairs to store filenames and their corresponding median temperature
    std::vector<std::pair<std::string, double>> fileTempPairs;

    for (const auto& filename : filenames)
    {
        double median = filename_to_median(filename);
        fileTempPairs.emplace_back(filename, median);
    }

    // Sort the vector based on the median temperature
    std::sort(fileTempPairs.begin(), fileTempPairs.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second; // Hottest to coldest
        });

    // Ensure destination folder exists or clear it if it does
    clear_folder(destination_folder);
    fs::create_directories(destination_folder);

    // Copy files to the destination folder in the new order
    int index = 1;
    for (const auto& pair : fileTempPairs)
    {
        fs::path source_file = pair.first;
        fs::path dest_file = fs::path(destination_folder) / (std::to_string(index) + source_file.extension().string());
        try
        {
            fs::copy_file(source_file, dest_file, fs::copy_options::overwrite_existing);
            ++index;
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Failed to copy file: " << source_file << " -> " << e.what() << std::endl;
        }
    }
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // folder to load images
    const char* image_folder = "images/unsorted";
    const char* sorted_folder = "cw2_cpu_sorted";

    if (!fs::is_directory(image_folder))
    {
        printf("Directory \"%s\" not found: please make sure it exists, and if it's a relative path, it's under your WORKING directory\n", image_folder);
        return -1;
    }

    std::vector<std::string> imageFilenames;
    for (auto& p : fs::directory_iterator(image_folder))
        imageFilenames.push_back(p.path().string());

    // Sort and save images to the `cw1_cpu` folder
    sort_and_save_images(imageFilenames, sorted_folder);

    std::cout << "Images have been sorted and saved to the \"" << sorted_folder << "\" folder.\n";

    return EXIT_SUCCESS;
}
