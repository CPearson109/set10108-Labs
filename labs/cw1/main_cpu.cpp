#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <set>
#include <limits>

// Function to read the entire content of a file into a vector<char>
std::vector<char> read_file(const char* filename)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Move the file cursor to the end of the file to get its size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    // Return the file cursor to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Create a vector of the same size as the file to hold the content
    std::vector<char> buffer(fileSize);

    // Read the entire file into the vector
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    // Close the file
    file.close();

    // Output the number of bytes read
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    // Convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](unsigned char c) { return std::tolower(c); });

    return buffer;
}

// Function to calculate the occurrences of a token in the data
int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i < int(data.size()); ++i)
    {
        // Test 1: Does this match the token?
        if (i + tokenLen > int(data.size()))
            break;

        if (strncmp(&data[i], token, tokenLen) != 0)
            continue;

        // Test 2: Is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && std::isalpha(static_cast<unsigned char>(data[iPrefix])))
            continue;

        // Test 3: Is the suffix a non-letter character?
        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && std::isalpha(static_cast<unsigned char>(data[iSuffix])))
            continue;

        ++numOccurrences;
    }
    return numOccurrences;
}

// Function to convert a string to lowercase
std::string to_lowercase(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

int main()
{

    // Define dataset folder
    std::string dataset_folder = "dataset";

    // List all files in the dataset folder
    std::vector<std::string> file_list;
    for (const auto& entry : std::filesystem::directory_iterator(dataset_folder)) {
        if (entry.is_regular_file()) {
            file_list.push_back(entry.path().filename().string());
        }
    }

    // Output the list of files
    std::cout << "Available files in " << dataset_folder << ":\n";
    for (const auto& filename : file_list) {
        std::cout << filename << "\n";
    }

    // Ask user to select a file
    std::cout << "Please enter the name of the file you wish to select: ";
    std::string selected_file;
    std::getline(std::cin, selected_file);

    // Check if the selected file exists in the dataset folder
    std::set<std::string> file_set(file_list.begin(), file_list.end());
    if (file_set.find(selected_file) == file_set.end()) {
        std::cerr << "Error: File not found in the dataset folder.\n";
        return -1;
    }

    // Construct the file path
    std::string filepath = dataset_folder + "/" + selected_file;

    // Read the file
    std::vector<char> file_data = read_file(filepath.c_str());
    if (file_data.empty())
        return -1;

    // Ask the user to type in the word(s) to search for
    std::cout << "Please enter the word(s) you wish to search for, separated by spaces: ";
    std::string input_line;
    std::getline(std::cin, input_line);

    // Parse the words and convert them to lowercase
    std::istringstream iss(input_line);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        words.push_back(to_lowercase(word));
    }

    // For each word, calculate occurrences and measure time
    for (const auto& word : words) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int occurrences = calc_token_occurrences(file_data, word.c_str());
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        std::cout << "Found " << occurrences << " occurrences of word: " << word
            << " in " << duration.count()*1000 << " milliseconds." << std::endl;
    }

    // Wait for user input before closing
    std::cout << "Press Enter to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return 0;
}
